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
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS


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
                 loss_weight=1.0,
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
        self.loss_weight = loss_weight

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
        llm = AutoModelForCausalLM.from_pretrained(
            self.llm_name, torch_dtype=torch.bfloat16)
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
        """Per-agent query [N, C]: prefer motion track_query [1,N,C], else
        track_query_embeddings [N,C]. Returns None when unavailable."""
        if outs_motion and outs_motion.get('track_query') is not None:
            q = outs_motion['track_query']
            return q[0] if q.dim() == 3 else q
        emb = (outs_track or {}).get('track_query_embeddings')
        return emb if emb is not None else None

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

    def forward_train(self, outs_motion, outs_track=None, gt_caption=None):
        """LM loss over [prompt | object tokens | caption].

        Supervises only the caption span. Returns dict(loss_llm=...). When
        there is no caption or no agent query, returns a graph-connected zero
        so DDP/backward stays happy.
        """
        agent_query = self._agent_query(outs_motion, outs_track)
        device = (agent_query.device if agent_query is not None
                  else next(self.projector.parameters()).device)

        target = gt_caption[0] if isinstance(gt_caption, (list, tuple)) \
            else gt_caption
        if not target or agent_query is None or agent_query.numel() == 0:
            zero = self.projector[0].weight.sum() * 0.0
            return dict(loss_llm=zero)

        centres = self._agent_centres(outs_track)
        obj_tok = self._object_tokens(agent_query, centres, device)  # [n,d]
        prompt_ids, prompt_emb = self._embed_text(self.prompt, device)
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
        device = (agent_query.device if agent_query is not None
                  else next(self.projector.parameters()).device)

        prompt_ids, prompt_emb = self._embed_text(self.prompt, device)
        if agent_query is None or agent_query.numel() == 0:
            inputs_embeds = prompt_emb
        else:
            centres = self._agent_centres(outs_track)
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


