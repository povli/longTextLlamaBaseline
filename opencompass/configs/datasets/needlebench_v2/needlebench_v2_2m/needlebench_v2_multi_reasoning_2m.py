from opencompass.datasets.needlebench_v2.multi import NeedleBenchMultiDataset
from mmengine.config import read_base

with read_base():
    from .needlebench_v2_single_2m import (
        depths_list,
        context_lengths,
        needlebench_reader_cfg,
        needlebench_infer_cfg,
    )
    from opencompass.configs.datasets.needlebench_v2.atc.atc_0shot_nocot_2_power_en import (  # noqa: E501
        needlebench_atc_eval_cfg as needlebench_eval_cfg)


base_path = 'opencompass/needlebench'
diff = 10

# English
needlebench_2needle_en_datasets = []
needlebench_3needle_en_datasets = []
needlebench_4needle_en_datasets = []
needlebench_5needle_en_datasets = []

for num_needles in range(2, 6):
    for original_context_length in context_lengths:
        for depth_percent in depths_list:
            dataset_dict = {
                'abbr': f'Length{original_context_length}'
                f'Depth{int(depth_percent)}_{num_needles}needle_en_2m',
                'type': NeedleBenchMultiDataset,
                'path': base_path,
                'length': original_context_length,
                'depth': int(depth_percent),
                'tokenizer_model': 'gpt-4',
                'file_list': ['PaulGrahamEssays.jsonl'],
                'num_repeats_per_file': 2,
                'length_buffer': 3000,
                'language': 'English',
                'needle_file_name': 'names.json',
                'num_needles': num_needles,
                'diff': diff,
                'reader_cfg': needlebench_reader_cfg,
                'infer_cfg': needlebench_infer_cfg,
                'eval_cfg': needlebench_eval_cfg,
            }
            globals()[f'needlebench_{num_needles}needle_en_datasets'].append(
                dataset_dict)

# Chinese
needlebench_2needle_zh_datasets = []
needlebench_3needle_zh_datasets = []
needlebench_4needle_zh_datasets = []
needlebench_5needle_zh_datasets = []

for num_needles in range(2, 6):
    for original_context_length in context_lengths:
        for depth_percent in depths_list:
            dataset_dict = {
                'abbr': f'Length{original_context_length}'
                f'Depth{int(depth_percent)}_{num_needles}needle_zh_2m',
                'type': NeedleBenchMultiDataset,
                'path': base_path,
                'length': original_context_length,
                'depth': int(depth_percent),
                'tokenizer_model': 'gpt-4',
                'file_list': ['zh_finance.jsonl'],
                'num_repeats_per_file': 2,
                'length_buffer': 200,
                'language': 'Chinese',
                'needle_file_name': 'names.json',
                'num_needles': num_needles,
                'diff': diff,
                'reader_cfg': needlebench_reader_cfg,
                'infer_cfg': needlebench_infer_cfg,
                'eval_cfg': needlebench_eval_cfg,
            }
            globals()[f'needlebench_{num_needles}needle_zh_datasets'].append(
                dataset_dict)
