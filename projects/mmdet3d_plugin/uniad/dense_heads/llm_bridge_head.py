"""LLMBridgeHead: distill image-only scene semantics into LiDAR queries.

See documents/llm_integration_plan.md. This head projects UniAD's
object-centric LiDAR queries (track_query) into the token space of a small
causal LLM (Qwen2.5-0.5B) and trains it, via a language-modelling loss, to
reproduce the VLM teacher's Chinese scene summary. At inference it
generates a caption from LiDAR queries alone -- no image is ever used by the
model, so this branch does not touch the TensorRT LiDAR inference path.

Training/offline only. transformers/peft are imported lazily so the rest of
the detector still builds in environments without an LLM. The backbone LLM is
frozen; only the projector (+ optional LoRA adapters) train.

IMPORTANT (verified on uniad_train, torch1.12 + transformers 4.46.3):
  - Qwen2.5-0.5B must run in bf16 (or fp32); fp16 yields NaN LM loss.
"""
import json
import os

import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS


def _is_qwen3_checkpoint(llm_name):
    """Return True for local/HF Qwen3 checkpoints without importing HF code."""
    name = str(llm_name)
    cfg_path = os.path.join(name, 'config.json')
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, 'r') as f:
                return json.load(f).get('model_type') == 'qwen3'
        except (OSError, ValueError):
            pass
    return 'qwen3' in name.lower()


@HEADS.register_module()
class LLMBridgeHead(BaseModule):
    """Project per-agent LiDAR queries into an LLM and supervise with text."""

    IGNORE_INDEX = -100

    def __init__(self,
                 llm_name='/mnt/disk1/models/Qwen2.5-0.5B-Instruct',
                 d_llm=896,
                 in_channels=256,
                 max_agents=64,
                 use_spatial_pe=True,
                 pc_range=None,
                 freeze_llm=True,
                 use_lora=True,
                 lora_cfg=None,
                 max_text_len=128,
                 prompt='请用一句话描述本车周围的港口场景。',
                 qa_prompt_template='请回答问题：{question}',
                 training_mode='caption',
                 qa_answer_types=None,
                 qa_index_strategy='first',
                 loss_weight=1.0,
                 detach_inputs=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.llm_name = llm_name
        self.d_llm = d_llm
        self.in_channels = in_channels
        self.max_agents = max_agents
        self.use_spatial_pe = use_spatial_pe
        self.pc_range = pc_range
        self.freeze_llm = freeze_llm
        self.use_lora = use_lora
        self.lora_cfg = lora_cfg or dict(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=['q_proj', 'v_proj'])
        self.max_text_len = max_text_len
        self.prompt = prompt
        self.qa_prompt_template = qa_prompt_template
        self.training_mode = training_mode
        if self.training_mode not in ('caption', 'qa'):
            raise ValueError(
                f'Unsupported LLMBridgeHead training_mode={training_mode!r}; '
                'use "caption" or "qa".')
        self.qa_answer_types = (
            None if qa_answer_types is None else set(qa_answer_types))
        self.qa_index_strategy = qa_index_strategy
        if self.qa_index_strategy not in ('first', 'hash'):
            raise ValueError(
                f'Unsupported qa_index_strategy={qa_index_strategy!r}; '
                'use "first" or "hash".')
        self.loss_weight = loss_weight
        self.detach_inputs = detach_inputs

        self.projector = nn.Sequential(
            nn.Linear(in_channels, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
        if use_spatial_pe:
            self.spatial_pe = nn.Sequential(
                nn.Linear(3, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))

        # Build the LLM (+LoRA) eagerly here, NOT lazily on first forward.
        # The training loop builds the optimizer and wraps DDP right after the
        # model is constructed and before any forward, so a lazily-created LLM
        # would leave its LoRA params out of the optimizer (never trained) and
        # unregistered by DDP (no grad sync). Building in __init__ makes the
        # params visible to both. They live on CPU until the trainer's
        # model.cuda() moves the whole module.
        self.tokenizer = None
        self._llm = None
        self._build_llm()

    def _build_llm(self):
        """Construct tokenizer + (LoRA-wrapped, bf16) causal LM in __init__."""
        is_qwen3 = _is_qwen3_checkpoint(self.llm_name)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                'LLMBridgeHead needs transformers (and peft for LoRA). In '
                'uniad_train: pip install transformers==4.46.3 peft==0.13.2 '
                '(use --no-deps for torch so mmcv.ops stays intact).') from e
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # bf16 is mandatory: fp16 gives NaN loss on this model (see module doc).
        model_kwargs = dict(torch_dtype=torch.bfloat16)
        if is_qwen3 and not hasattr(
                torch.nn.functional, 'scaled_dot_product_attention'):
            model_kwargs['attn_implementation'] = 'eager'
        llm = AutoModelForCausalLM.from_pretrained(
            self.llm_name, **model_kwargs)
        if self.freeze_llm:
            llm.eval()
            for p in llm.parameters():
                p.requires_grad = False
        if self.use_lora:
            from peft import LoraConfig, get_peft_model
            llm = get_peft_model(
                llm, LoraConfig(task_type='CAUSAL_LM', **self.lora_cfg))
        # Register as a submodule so optimizer/DDP/.cuda()/.train() see it.
        self._llm = llm

    @property
    def _llm_dtype(self):
        return self._llm.get_input_embeddings().weight.dtype

    @staticmethod
    def _agent_query(outs_motion, outs_track):
        """Per-agent query [N, C], aligned 1:1 with _agent_centres.

        Use track_query_embeddings (NOT motion's track_query): MotionHeadLidar
        filters/reorders its track_query down to vehicle classes and strips the
        SDC slot, while the box centres come from the UNFILTERED
        track_bbox_results -- so motion query[i] and centre[i] would describe
        different objects. track_query_embeddings and track_bbox_results are
        built together in select_active_track_query (same topk bbox_index +
        same mask), so they are same-order, same-length. Using them also keeps
        non-vehicle agents (pedestrians, cones) that the caption may reference
        but motion drops. Falls back to motion track_query only if track
        embeddings are absent.
        """
        emb = (outs_track or {}).get('track_query_embeddings')
        if emb is not None:
            return emb
        if outs_motion and outs_motion.get('track_query') is not None:
            q = outs_motion['track_query']
            return q[0] if q.dim() == 3 else q
        return None

    @staticmethod
    def _agent_centres(outs_track):
        """BEV box centres [N,3] from track_bbox_results, or None."""
        tbr = (outs_track or {}).get('track_bbox_results')
        if not tbr:
            return None
        boxes_3d = tbr[0][0]
        return getattr(boxes_3d, 'gravity_center', None)

    def _object_tokens(self, agent_query, centres, device):
        """[N,C] queries (+centres) -> [n, d_llm] object tokens (n<=max_agents).

        The projector/spatial_pe run in their own (fp32) dtype for stable
        training; the result is cast to the LLM's dtype (bf16) before it joins
        the token stream.
        """
        n = min(agent_query.size(0), self.max_agents)
        proj_dtype = self.projector[0].weight.dtype
        if self.detach_inputs:
            agent_query = agent_query.detach()
            if centres is not None:
                centres = centres.detach()
        q = agent_query[:n].to(device=device, dtype=proj_dtype)
        tokens = self.projector(q)
        if self.use_spatial_pe and centres is not None and n > 0:
            c = centres[:n].to(device=device, dtype=proj_dtype)
            tokens = tokens + self.spatial_pe(c)
        return tokens.to(self._llm_dtype)

    def _embed_text(self, text, device):
        """text -> (ids[1,L], embeds[1,L,d_llm])."""
        ids = self.tokenizer(
            text, return_tensors='pt', truncation=True,
            max_length=self.max_text_len).input_ids.to(device)
        return ids, self._llm.get_input_embeddings()(ids)

    @staticmethod
    def _unwrap_single(value):
        while (isinstance(value, (list, tuple)) and len(value) == 1 and
               not (value and isinstance(value[0], dict))):
            value = value[0]
        return value

    def _select_qa(self, gt_qa):
        """Pick one QA dict from current-frame QA metadata."""
        gt_qa = self._unwrap_single(gt_qa)
        if not gt_qa:
            return None
        if isinstance(gt_qa, dict):
            qa_list = [gt_qa]
        else:
            qa_list = list(gt_qa)
        if self.qa_answer_types is not None:
            qa_list = [
                qa for qa in qa_list
                if isinstance(qa, dict)
                and qa.get('answer_type') in self.qa_answer_types
            ]
        else:
            qa_list = [qa for qa in qa_list if isinstance(qa, dict)]
        if not qa_list:
            return None
        if self.qa_index_strategy == 'hash':
            # Deterministic per-frame spread without importing random state.
            key = str(qa_list[0].get('id', ''))
            idx = sum(ord(ch) for ch in key) % len(qa_list)
            return qa_list[idx]
        return qa_list[0]

    def _training_texts(self, gt_caption=None, gt_qa=None):
        """Return (prompt, target) for the configured training mode."""
        if self.training_mode == 'qa':
            qa = self._select_qa(gt_qa)
            if qa is None:
                return None, None
            question = qa.get('question')
            answer = qa.get('answer')
            if not question or not answer:
                return None, None
            prompt = self.qa_prompt_template.format(
                question=question, answer_type=qa.get('answer_type', ''),
                gt_source=qa.get('gt_source', ''))
            return prompt, answer

        target = self._unwrap_single(gt_caption)
        if not target:
            return None, None
        return self.prompt, target

    def forward_train(self, outs_motion, outs_track=None, gt_caption=None,
                      gt_qa=None):
        """LM loss over [prompt | object tokens | target text].

        In caption mode target text is the VLM scene summary. In QA mode target
        text is one deterministic hard-GT answer selected from gt_qa and the
        prompt contains that question. Supervises only the target span. When
        target text or agent query is absent, returns a graph-connected zero so
        DDP/backward stays happy.
        """
        agent_query = self._agent_query(outs_motion, outs_track)
        device = (agent_query.device if agent_query is not None
                  else next(self.projector.parameters()).device)

        prompt, target = self._training_texts(
            gt_caption=gt_caption, gt_qa=gt_qa)
        if not target or agent_query is None or agent_query.numel() == 0:
            zero = self.projector[0].weight.sum() * 0.0
            return dict(loss_llm=zero)

        centres = self._agent_centres(outs_track)
        obj_tok = self._object_tokens(agent_query, centres, device)  # [n,d]
        prompt_ids, prompt_emb = self._embed_text(prompt, device)
        tgt_ids, tgt_emb = self._embed_text(target, device)

        obj_emb = obj_tok.unsqueeze(0)  # [1,n,d]
        inputs_embeds = torch.cat([prompt_emb, obj_emb, tgt_emb], dim=1)
        n_prefix = prompt_emb.size(1) + obj_emb.size(1)
        labels = torch.cat([
            torch.full((1, n_prefix), self.IGNORE_INDEX,
                       device=device, dtype=torch.long),
            tgt_ids], dim=1)
        out = self._llm(inputs_embeds=inputs_embeds, labels=labels)
        return dict(loss_llm=out.loss * self.loss_weight)

    @torch.no_grad()
    def forward_test(self, outs_motion, outs_track=None, max_new_tokens=64):
        """Greedy-generate a caption from LiDAR object tokens. Returns str."""
        agent_query = self._agent_query(outs_motion, outs_track)
        centres = self._agent_centres(outs_track)
        return self.generate_from_query(
            agent_query, centres, max_new_tokens=max_new_tokens)

    def generate_from_query(self, agent_query, centres, max_new_tokens=64):
        """Greedy caption from explicit (agent_query[N,C], centres[N,3]).

        The seam used by eval baselines (eval_llm_caption.py): pass agent_query
        =None for the no-query language-prior baseline, or pass another frame's
        query/centres for the shuffle control. forward_test routes the normal
        per-frame query/centres through here.
        """
        device = (agent_query.device if agent_query is not None
                  else next(self.projector.parameters()).device)

        prompt_ids, prompt_emb = self._embed_text(self.prompt, device)
        if agent_query is None or agent_query.numel() == 0:
            inputs_embeds = prompt_emb
        else:
            obj_tok = self._object_tokens(agent_query, centres, device)
            inputs_embeds = torch.cat([prompt_emb, obj_tok.unsqueeze(0)], dim=1)
        # In inputs_embeds mode the model can't infer the mask, and pad==eos
        # here, so pass an explicit all-ones mask + ids for reliable decoding.
        attn = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=device)
        gen = self._llm.generate(
            inputs_embeds=inputs_embeds, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.batch_decode(
            gen, skip_special_tokens=True)[0].strip()
