from mmengine.config import read_base
from opencompass.models import OpenAISDK
from opencompass.configs.summarizers.needlebench import needlebench_v2_1m_subset_summarizer as summarizer  # noqa: E501
import os

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='user'),
        dict(role='BOT', api_role='assistant', generate=True),
    ],
)

# Only keep these context lengths
subset_lengths = [580000, 720000, 1000000]
depths_list_10 = [i for i in range(0, 101, 10)]

with read_base():
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_1000k.needlebench_v2_1000k import needlebench_datasets  # noqa: E501

datasets = [
    d for d in needlebench_datasets if d.get('length') in subset_lengths
]

openai_base = os.getenv('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')
del os

models = [
    dict(
        abbr='llama-1.3b-vllm-api-1m-subset',
        type=OpenAISDK,
        path='huihui-ai/Llama-3.2-1B-Instruct-abliterated',
        key='EMPTY',
        openai_api_base=openai_base,
        max_seq_len=1048576,  # ensure vLLM started with >= this window
        max_out_len=256,
        temperature=0.0,
        meta_template=api_meta_template,
        batch_size=1,
        query_per_second=2,
        retry=3,
    )
]

work_dir = './outputs/needlebench_1m_subset'
