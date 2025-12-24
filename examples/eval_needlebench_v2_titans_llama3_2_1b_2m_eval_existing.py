from mmengine.config import read_base
from opencompass.models import HuggingFaceBaseModel
from pathlib import Path
import os
import re

# Keep memory/SMI logging consistent with the main config.
os.environ.setdefault('OPENCOMPASS_AUTO_SMI', '1')
os.environ.setdefault('OPENCOMPASS_SMI_INTERVAL', '1')
os.environ.setdefault('OPENCOMPASS_MEM_PATCH', '1')

# Override via env var if needed, default is non-LoRA Titans-style delta_product.
TPTT_SUBFOLDER = os.getenv('TPTT_SUBFOLDER', 'delta_product_m0.5_constant')
EVAL_REUSE_STAMP = os.getenv('EVAL_REUSE_STAMP', '')
del os

work_dir = './outputs/needlebench_2m_titans_llama3_2_1b'

with read_base():
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_2m.needlebench_v2_2m import (  # noqa: E501
        needlebench_datasets as datasets,
    )
    from opencompass.configs.summarizers.needlebench_v2_2m_summarizer import (  # noqa: E501
        needlebench_v2_2m_summarizer as summarizer,
    )

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr=f'titans-llama-3.2-1b-{TPTT_SUBFOLDER}',
        path='ffurfaro/Titans-Llama-3.2-1B',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            subfolder=TPTT_SUBFOLDER,
            torch_dtype='torch.bfloat16',
            max_self_attn_length=4096,
        ),
        tokenizer_kwargs=dict(
            trust_remote_code=True,
            subfolder=TPTT_SUBFOLDER,
            padding_side='left',
        ),
        generation_kwargs=dict(
            use_cache=False,
        ),
        max_seq_len=2048000,
        max_out_len=64,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]

# Evaluate only datasets that already have prediction files under the reuse stamp.
if EVAL_REUSE_STAMP:
    model_abbr = models[0]['abbr']
    pred_dir = Path(work_dir) / EVAL_REUSE_STAMP / 'predictions' / model_abbr
    if pred_dir.exists():
        pred_abbr = {
            re.sub(r'_\d+$', '', p.stem)
            for p in pred_dir.glob('*.json')
        }
        datasets = [d for d in datasets if d.get('abbr') in pred_abbr]
