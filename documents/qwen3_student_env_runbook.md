# Qwen3-0.6B Student 隔离环境手册

> 更新时间：2026-06-25。目标：在不升级主 `uniad_train` 环境的前提下，运行
> `projects/configs/stage2_e2e_lidar/base_e2e_lidar_llm_probe_qwen3_0p6b.py`。

## 为什么要单独隔离

Qwen3-0.6B 的 checkpoint 是 `model_type=qwen3`，需要 `transformers>=4.51`。
但主训练环境 `uniad_train` 锁在 UniAD 旧栈：

- `torch==1.12.1+cu116`
- `mmcv/mmcv-full` 与自定义 CUDA ops 按旧 torch 编译
- `spconv`、`mmdet3d`、`pytorch_lightning` 都有旧版本约束

直接在 `uniad_train` 升级 transformers/accelerate 容易把 torch 拉到 2.x，导致
`mmcv.ops`、`third_party/uniad_mmdet3d` CUDA 扩展失效。因此 Qwen3 student 走独立环境：

```bash
conda activate /mnt/disk1/conda_envs/uniad_train_qwen3_py39
```

不要直接 clone `uniad_train` 的 Python3.8 环境来升包：Qwen3 需要的 transformers 新版本要求
Python>=3.9，Python3.8 clone 会卡在包版本上。

## 当前已验证状态

环境位置：

```text
/mnt/disk1/conda_envs/uniad_train_qwen3_py39
```

关键版本：

```text
Python 3.9.25
torch 1.12.1+cu116
torchvision 0.13.1+cu116
numpy 1.22.4
mmcv-full 1.5.2
mmdet 2.25.1
mmsegmentation 0.29.1
mmdet3d 1.0.0rc4
transformers 4.51.3
peft 0.13.2
spconv-cu116 2.3.6
```

已验证：

```bash
PYTHONPATH=$(pwd) python - <<'PY'
from mmcv import Config
from third_party.uniad_mmdet3d.models.builder import build_model
import projects.mmdet3d_plugin

cfg = Config.fromfile(
    'projects/configs/stage2_e2e_lidar/base_e2e_lidar_llm_probe_qwen3_0p6b.py')
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
print(type(model).__name__, type(model.llm_head._llm).__name__)
print(round(sum(p.numel() for p in model.parameters()
                if p.requires_grad) / 1e6, 3))
PY
```

期望输出：

```text
UniADMotionLidar PeftModelForCausalLM
4.66
```

复核记录：2026-06-25 另建全新环境
`/mnt/disk1/conda_envs/uniad_train_qwen3_py39_doccheck`，按本文流程从零安装并完成
基础 import、`projects.mmdet3d_plugin` import、`LLMBridgeHead` loss smoke 和真实 config build。

## 从零重建命令

放在 `/mnt/disk1/conda_envs`，避免根盘空间压力：

```bash
mkdir -p /mnt/disk1/conda_envs
conda create -p /mnt/disk1/conda_envs/uniad_train_qwen3_py39 python=3.9 pip -y
conda activate /mnt/disk1/conda_envs/uniad_train_qwen3_py39
```

关闭用户 site-packages 污染：

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/no_user_site.sh" <<'SH'
export _OLD_PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-}"
export PYTHONNOUSERSITE=1
SH
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/no_user_site.sh" <<'SH'
if [ -n "${_OLD_PYTHONNOUSERSITE:-}" ]; then
  export PYTHONNOUSERSITE="${_OLD_PYTHONNOUSERSITE}"
else
  unset PYTHONNOUSERSITE
fi
unset _OLD_PYTHONNOUSERSITE
SH
conda deactivate
conda activate /mnt/disk1/conda_envs/uniad_train_qwen3_py39
```

安装 torch 旧栈：

```bash
pip install --no-cache-dir \
  'torch==1.12.1+cu116' \
  'torchvision==0.13.1+cu116' \
  'torchaudio==0.12.1' \
  --extra-index-url https://download.pytorch.org/whl/cu116
```

安装基础依赖。torch wheel 可能临时带入较新的 `numpy/Pillow`，这一步会固定回 UniAD
旧栈可用版本；这里不要用 `--no-deps`，否则 `matplotlib/tensorboard/numba` 的传递依赖会缺失：

```bash
pip install \
  'numpy==1.22.4' 'Pillow==9.5.0' 'packaging==26.2' \
  'PyYAML==6.0.3' 'six==1.17.0' 'terminaltables==3.1.10' \
  'prettytable==3.16.0' 'tqdm==4.68.3' 'einops==0.8.2' \
  'shapely==2.0.7' 'pyquaternion==0.9.9' \
  'scipy==1.10.1' 'opencv-python==4.7.0.72' 'matplotlib==3.5.3' \
  'pandas==1.4.4' 'numba==0.58.1' 'protobuf==5.29.6' \
  'tensorboard==2.14.0'
```

安装 OpenMMLab 旧栈。`mmcv-full` 建议用 `--no-build-isolation`，避免临时构建环境缺 torch/pkg_resources：

```bash
pip install --no-deps --no-build-isolation 'mmcv-full==1.5.2'
pip install --no-deps 'addict==2.4.0' 'pycocotools==2.0.7' \
  'platformdirs==4.3.6'
pip install --no-deps 'mmdet==2.25.1' 'mmsegmentation==0.29.1' \
  'mmcls==0.25.0' 'yapf==0.40.1' 'tomli==2.2.1'
pip install --no-deps --no-build-isolation 'mmdet3d==1.0.0rc4'
```

安装 UniAD import 链和 spconv 依赖：

```bash
pip install --no-deps \
  'trimesh==4.4.9' 'plyfile==1.0.3' 'scikit-image==0.19.3' \
  'networkx==2.8.8' 'motmetrics==1.1.3' 'casadi==3.6.7' \
  'pytorch-lightning==1.2.5' 'torchmetrics==0.11.4' \
  'pyDeprecate==0.3.2' 'imageio==2.34.2' 'PyWavelets==1.4.1' \
  'tifffile==2023.7.10' 'lyft-dataset-sdk==0.0.8' \
  'spconv-cu116==2.3.6' 'cumm-cu116==0.4.11' 'pccm==0.4.16' \
  'ccimport==0.4.4' 'fire==0.6.0' 'pybind11==2.13.6' \
  'ninja==1.11.1.4' 'portalocker==2.10.1' 'future==1.0.0' \
  'fsspec==2024.6.1' 'lark==1.1.9' 'termcolor==2.5.0' \
  'nuscenes-devkit==1.1.9' 'scikit-learn==1.3.2' \
  'joblib==1.4.2' 'threadpoolctl==3.5.0' 'cachetools==5.5.0' \
  'descartes==1.1.0'
pip install 'IPython==8.18.1'
```

安装 Qwen3 student 所需 LLM 依赖，必须 `--no-deps`，避免 pip 升级 torch：

```bash
pip install --no-deps \
  'transformers==4.51.3' 'peft==0.13.2' 'accelerate==1.1.1' \
  'tokenizers==0.21.4' 'safetensors==0.6.2' 'huggingface-hub==0.33.5' \
  'filelock==3.18.0' 'regex==2024.11.6' 'hf-xet==1.1.5' \
  'psutil==6.1.1'
```

编译仓库自定义 CUDA ops：

```bash
cd /home/baojiali/Downloads/public_code/DL4AGX/AV-Solutions/uniad-trt/UniAD_train/UniAD
(
  cd third_party/uniad_mmdet3d
  FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='8.6' python setup.py build_ext --inplace
)
```

## sitecustomize.py 兼容层

`transformers>=4.51` 在 import 阶段会引用 torch2-only 符号：

- `torch.distributed.tensor`
- `torch.float8_e4m3fn` / `torch.float8_e5m2`
- `torch.compiler`
- `torch._dynamo`
- `Module.named_parameters(remove_duplicate=...)`
- `Module.load_state_dict(assign=True)`

这些符号在 torch1.12 中不存在，但当前 Qwen3 probe 只走 eager、单进程、非 tensor-parallel 路径。
因此这个隔离环境用 `sitecustomize.py` 在 Python 启动时补最小 shim：

```text
/mnt/disk1/conda_envs/uniad_train_qwen3_py39/lib/python3.9/site-packages/sitecustomize.py
```

注意：这个文件是环境内文件，不在 git 里；主文档记录它的存在，是为了提醒不要把主 `uniad_train`
直接升级到 transformers4.51+。

从零重建时写入：

```bash
cat > "$CONDA_PREFIX/lib/python3.9/site-packages/sitecustomize.py" <<'PY'
"""Compatibility shims for the isolated UniAD Qwen3 student environment."""
import inspect
import sys
import types
from collections import namedtuple

try:
    import torch
except Exception:
    torch = None


def _install():
    if torch is None:
        return
    try:
        torch_major = int(torch.__version__.split('+', 1)[0].split('.', 1)[0])
    except (TypeError, ValueError):
        torch_major = 2
    if torch_major >= 2:
        return

    if not hasattr(torch, 'float8_e4m3fn'):
        torch.float8_e4m3fn = torch.float16
    if not hasattr(torch, 'float8_e5m2'):
        torch.float8_e5m2 = torch.float16

    if not hasattr(torch, 'compiler'):
        def _disable(fn=None, recursive=True):
            return fn if fn is not None else (lambda f: f)

        torch.compiler = types.SimpleNamespace(
            is_compiling=lambda: False,
            is_exporting=lambda: False,
            disable=_disable)

    if not hasattr(torch, '_dynamo'):
        torch._dynamo = types.SimpleNamespace(
            is_compiling=lambda: False,
            mark_static_address=lambda *args, **kwargs: None,
            config=types.SimpleNamespace(cache_size_limit=64))

    if 'remove_duplicate' not in inspect.signature(
            torch.nn.Module.named_parameters).parameters:
        orig_named_parameters = torch.nn.Module.named_parameters

        def named_parameters(self, prefix='', recurse=True,
                             remove_duplicate=True):
            return orig_named_parameters(
                self, prefix=prefix, recurse=recurse)

        torch.nn.Module.named_parameters = named_parameters

    if 'assign' not in inspect.signature(
            torch.nn.Module.load_state_dict).parameters:
        orig_load_state_dict = torch.nn.Module.load_state_dict
        incompatible_keys = namedtuple(
            '_IncompatibleKeys', ['missing_keys', 'unexpected_keys'])

        def load_state_dict(self, state_dict, strict=True, assign=False):
            if not assign:
                return orig_load_state_dict(
                    self, state_dict, strict=strict)
            unexpected = []
            for key, tensor in state_dict.items():
                module = self
                parts = key.split('.')
                for part in parts[:-1]:
                    module = getattr(module, part)
                name = parts[-1]
                if name in module._parameters:
                    old = module._parameters[name]
                    requires_grad = True if old is None else old.requires_grad
                    module._parameters[name] = torch.nn.Parameter(
                        tensor.detach(), requires_grad=requires_grad)
                elif name in module._buffers:
                    module._buffers[name] = tensor.detach()
                else:
                    unexpected.append(key)
            if strict and unexpected:
                raise RuntimeError(
                    'Unexpected key(s): {}'.format(', '.join(unexpected)))
            return incompatible_keys([], unexpected)

        torch.nn.Module.load_state_dict = load_state_dict

    if 'torch.distributed.tensor' not in sys.modules:
        tensor_mod = types.ModuleType('torch.distributed.tensor')

        class Placement:
            pass

        class Replicate(Placement):
            pass

        class Shard(Placement):
            def __init__(self, dim=None):
                self.dim = dim

        class DTensor:
            @classmethod
            def from_local(cls, *args, **kwargs):
                raise RuntimeError(
                    'DTensor shim is import-only; tensor parallel execution '
                    'is not supported in the torch1.12 Qwen3 env.')

        tensor_mod.DTensor = DTensor
        tensor_mod.Placement = Placement
        tensor_mod.Replicate = Replicate
        tensor_mod.Shard = Shard
        sys.modules['torch.distributed.tensor'] = tensor_mod
        if hasattr(torch, 'distributed'):
            torch.distributed.tensor = tensor_mod


_install()
PY
```

## 验证命令

基础 import：

```bash
conda activate /mnt/disk1/conda_envs/uniad_train_qwen3_py39
python - <<'PY'
import sys, os, site
import torch, transformers, peft, mmcv, mmdet, mmseg, mmdet3d
import spconv.pytorch, nuscenes, IPython
print(sys.executable)
print(os.environ.get('PYTHONNOUSERSITE'), site.ENABLE_USER_SITE)
print(torch.__version__, transformers.__version__, peft.__version__, mmcv.__version__)
PY
```

LLMBridgeHead loss smoke：

```bash
PYTHONPATH=$(pwd) python - <<'PY'
import torch
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.uniad.dense_heads.llm_bridge_head import LLMBridgeHead

head = LLMBridgeHead(
    llm_name='/mnt/disk1/models/Qwen3-0.6B',
    d_llm=1024,
    in_channels=256,
    max_agents=4,
    freeze_llm=True,
    use_lora=True,
    detach_inputs=True,
    max_text_len=64)
outs_track = {'track_query_embeddings': torch.randn(3, 256),
              'track_bbox_results': []}
loss = head.forward_train(
    {}, outs_track, '左前方有集装箱卡车，本车应减速让行。')
print(float(loss['loss_llm'].detach().cpu()))
PY
```

真实 config build：

```bash
PYTHONPATH=$(pwd) python - <<'PY'
from mmcv import Config
from third_party.uniad_mmdet3d.models.builder import build_model
import projects.mmdet3d_plugin

cfg = Config.fromfile(
    'projects/configs/stage2_e2e_lidar/base_e2e_lidar_llm_probe_qwen3_0p6b.py')
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
print(type(model).__name__, type(model.llm_head._llm).__name__)
print(round(sum(p.numel() for p in model.parameters()
                if p.requires_grad) / 1e6, 3))
PY
```

## 已知 pip check 警告

当前环境 `pip check` 仍会提示几类非阻塞问题：

- `lyft-dataset-sdk` 缺 `black/flake8/plotly/pytest`：这些是开发/评估附属依赖，当前 build smoke 不需要。
- `nuscenes-devkit` 缺 `jupyter`：它是 notebook 附属依赖，当前 UniAD import/build 不需要。
- `mmdet3d` 声明要求 `networkx<2.3`、`numba==0.53.0`、旧 `trimesh`：Python3.9 下这些旧版本不现实；
  当前 UniAD build/import 已验证通过。
- `peft 0.13.2` metadata 要求 `torch>=1.13.0`：实测 Qwen3-0.6B LoRA head build 与 loss smoke
  在 torch1.12.1+cu116 下可跑。后续若出现 PEFT 深层 API 问题，再考虑退成 `use_lora=False`
  或迁移到 torch2 独立训练栈。

另一个容易踩的点：不要额外安装 `deprecate` 包，它会覆盖 `pyDeprecate` 提供的同名 `deprecate`
模块，导致 `pytorch_lightning.metrics` 旧 import 链缺 `void`。如果误装了：

```bash
pip uninstall -y deprecate
pip install --force-reinstall --no-deps 'pyDeprecate==0.3.2'
```

## 使用边界

- 这个环境只用于 Qwen3-0.6B student probe，不替代主 `uniad_train`。
- 不要在主 `uniad_train` 里直接安装 transformers4.51+。
- 若要跑正式训练，先用该环境跑 50-200 iter 小训练，确认 loss 有限、显存稳定，再上多卡。
