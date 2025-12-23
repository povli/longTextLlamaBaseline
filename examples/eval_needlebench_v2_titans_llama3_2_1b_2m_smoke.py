from mmengine.config import read_base
from opencompass.models import HuggingFaceBaseModel
import os

# Override via env var if needed, default is non-LoRA Titans-style delta_product.
TPTT_SUBFOLDER = os.getenv('TPTT_SUBFOLDER', 'delta_product_m0.5_constant')
BASE_MODEL_NAME = os.getenv('BASE_MODEL_NAME')
BASE_MODEL_SUBFOLDER = os.getenv('BASE_MODEL_SUBFOLDER')
BASE_TOKENIZER_PATH = os.getenv('BASE_TOKENIZER_PATH')
SMOKE_CONTEXT_LEN = int(os.getenv('SMOKE_CONTEXT_LEN', '2000000'))
SMOKE_DEPTH = int(os.getenv('SMOKE_DEPTH', '50'))
del os

base_model_overrides = {}
if BASE_MODEL_NAME:
    base_model_overrides['base_model_name'] = BASE_MODEL_NAME
if BASE_MODEL_SUBFOLDER:
    base_model_overrides['base_model_subfolder'] = BASE_MODEL_SUBFOLDER

tokenizer_path = BASE_TOKENIZER_PATH or BASE_MODEL_NAME
tokenizer_kwargs = dict(
    trust_remote_code=True,
    padding_side='left',
)
if tokenizer_path is None:
    tokenizer_kwargs['subfolder'] = TPTT_SUBFOLDER


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
        abbr=f'titans-llama-3.2-1b-{TPTT_SUBFOLDER}-smoke',
        path='ffurfaro/Titans-Llama-3.2-1B',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            subfolder=TPTT_SUBFOLDER,
            torch_dtype='torch.bfloat16',
            max_self_attn_length=4096,
            attn_implementation='flash_attention_2',
            **base_model_overrides,
        ),
        tokenizer_path=tokenizer_path,
        tokenizer_kwargs=dict(
            **tokenizer_kwargs,
        ),
        max_seq_len=2048000,
        max_out_len=256,
        batch_size=1,
        run_cfg=dict(num_gpus=2),
    )
]

work_dir = './outputs/needlebench_2m_titans_llama3_2_1b_smoke'
