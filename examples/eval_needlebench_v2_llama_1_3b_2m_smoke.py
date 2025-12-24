from mmengine.config import read_base
from opencompass.models import HuggingFaceBaseModel
import os

# Memory and SMI logging for reproducible runs.
os.environ.setdefault('OPENCOMPASS_AUTO_SMI', '1')
os.environ.setdefault('OPENCOMPASS_SMI_INTERVAL', '1')
os.environ.setdefault('OPENCOMPASS_MEM_PATCH', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# Override to point at your non-Titans Llama 1.3B checkpoint.
LLAMA_1_3B_PATH = os.getenv('LLAMA_1_3B_PATH', 'meta-llama/Llama-3.2-1B')
ATTN_IMPL = os.getenv('HF_ATTN_IMPL', 'sdpa')
SMOKE_CONTEXT_LEN = int(os.getenv('SMOKE_CONTEXT_LEN', '2000000'))
SMOKE_DEPTH = int(os.getenv('SMOKE_DEPTH', '60'))
del os

with read_base():
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_2m.needlebench_v2_single_2m import (  # noqa: E501
        needlebench_en_datasets,
    )

# Leave summarizer empty for smoke test to avoid lazy config serialization.
summarizer = {}

_dataset = None
for item in needlebench_en_datasets:
    if item.get('length') == SMOKE_CONTEXT_LEN and item.get('depth') == SMOKE_DEPTH:
        _dataset = item
        break
if _dataset is None:
    _dataset = needlebench_en_datasets[0]
_dataset = {**_dataset}
_dataset['abbr'] = f'{_dataset["abbr"]}_smoke'
_dataset['num_repeats_per_file'] = 1
datasets = [_dataset]

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-1.3b-2m-smoke',
        path=LLAMA_1_3B_PATH,
        model_kwargs=dict(
            device_map='auto',
            max_memory={0: '80GiB', 1: '80GiB'},
            torch_dtype='torch.bfloat16',
            attn_implementation=ATTN_IMPL,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
        ),
        generation_kwargs=dict(
            use_cache=False,
        ),
        max_seq_len=2048000,
        max_out_len=64,
        batch_size=1,
        run_cfg=dict(num_gpus=2),
    )
]

work_dir = './outputs/needlebench_2m_llama_1_3b_smoke'
