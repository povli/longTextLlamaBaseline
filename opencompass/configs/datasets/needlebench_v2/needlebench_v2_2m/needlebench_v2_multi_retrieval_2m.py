from opencompass.datasets.needlebench_v2.parallel import (
    NeedleBenchParallelDataset,
    NeedleBenchParallelEvaluator,
)
from mmengine.config import read_base

with read_base():
    from .needlebench_v2_single_2m import (
        depths_list as depths,
        context_lengths,
        needlebench_reader_cfg,
        needlebench_infer_cfg,
        needlebench_eval_cfg,
    )

needlebench_eval_cfg['evaluator']['type'] = NeedleBenchParallelEvaluator

base_path = 'opencompass/needlebench'
needle_file_name = 'needles.jsonl'

language_configs = [
    {
        'file_list': ['PaulGrahamEssays.jsonl'],
        'dataset_var': 'needlebench_en_datasets',
        'language': 'English',
        'length_buffer': 3000,
        'suffix': 'en',
    },
    {
        'file_list': ['zh_finance.jsonl'],
        'dataset_var': 'needlebench_zh_datasets',
        'language': 'Chinese',
        'length_buffer': 200,
        'suffix': 'zh',
    },
]

needlebench_en_datasets = []
needlebench_zh_datasets = []

for config in language_configs:
    for original_context_length in context_lengths:
        dataset_dict = {
            'abbr': f'Length{original_context_length}_parallel_{config["suffix"]}_2m',
            'type': NeedleBenchParallelDataset,
            'path': base_path,
            'needle_file_name': needle_file_name,
            'length': original_context_length,
            'depths': depths,
            'tokenizer_model': 'gpt-4',
            'file_list': config['file_list'],
            'num_repeats_per_file': 5,
            'length_buffer': config['length_buffer'],
            'language': config['language'],
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg,
        }
        globals()[config['dataset_var']].append(dataset_dict)
