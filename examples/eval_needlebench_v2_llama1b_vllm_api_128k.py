from mmengine.config import read_base
from opencompass.models import OpenAISDK
import os

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='user'),
        dict(role='BOT', api_role='assistant', generate=True),
    ],
)

with read_base():
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_128k import needlebench_datasets as datasets  # noqa: E501
    from opencompass.configs.summarizers.needlebench import needlebench_v2_128k_summarizer as summarizer  # noqa: E501

openai_base = os.getenv('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')
del os

models = [
    dict(
        abbr='llama-1.3b-vllm-api-128k',
        type=OpenAISDK,
        path='huihui-ai/Llama-3.2-1B-Instruct-abliterated',
        key='EMPTY',
        openai_api_base=openai_base,
        max_seq_len=131072,
        max_out_len=256,
        temperature=0.0,
        meta_template=api_meta_template,
        batch_size=1,
        query_per_second=2,
        retry=3,
    )
]

work_dir = './outputs/needlebench_128k'
