from mmengine.config import read_base
from opencompass.models import HuggingFaceBaseModel
import os

# Override via env var if needed, default is non-LoRA Titans-style delta_product.
os.environ.setdefault('OPENCOMPASS_AUTO_SMI', '1')
os.environ.setdefault('OPENCOMPASS_SMI_INTERVAL', '1')
os.environ.setdefault('OPENCOMPASS_MEM_PATCH', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
TPTT_SUBFOLDER = os.getenv('TPTT_SUBFOLDER', 'delta_product_m0.5_constant')
del os

with read_base():
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_2m.needlebench_v2_2m import needlebench_datasets as datasets  # noqa: E501
    from opencompass.configs.summarizers.needlebench_v2_2m_summarizer import needlebench_v2_2m_summarizer as summarizer  # noqa: E501

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr=f'titans-llama-3.2-1b-{TPTT_SUBFOLDER}',
        path='ffurfaro/Titans-Llama-3.2-1B',
        model_kwargs=dict(
            device_map='auto',
            max_memory={0: '60GiB', 1: '80GiB'},
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
        max_seq_len=2048000,
        max_out_len=64,
        batch_size=1,
        run_cfg=dict(num_gpus=2),
    )
]

work_dir = './outputs/needlebench_2m_titans_llama3_2_1b'
