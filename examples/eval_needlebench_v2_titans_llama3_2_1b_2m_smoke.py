from mmengine.config import read_base
from opencompass.models import HuggingFaceBaseModel
import os
import torch

# Override via env var if needed, default is non-LoRA Titans-style delta_product.
TPTT_SUBFOLDER = os.getenv('TPTT_SUBFOLDER', 'delta_product_m0.5_constant')
SMOKE_CONTEXT_LEN = int(os.getenv('SMOKE_CONTEXT_LEN', '2000000'))
SMOKE_DEPTH = int(os.getenv('SMOKE_DEPTH', '50'))


def _wrap_generate_for_mem():
    orig_generate = HuggingFaceBaseModel.generate

    def wrapped(self, inputs, max_out_len, **kwargs):
        devices = []
        base_stats = {}
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
            for i in devices:
                torch.cuda.synchronize(i)
                base_stats[i] = (
                    torch.cuda.memory_allocated(i),
                    torch.cuda.memory_reserved(i),
                )
                torch.cuda.reset_peak_memory_stats(i)

        out = orig_generate(self, inputs, max_out_len, **kwargs)

        if devices:
            for i in devices:
                torch.cuda.synchronize(i)
            parts = []
            for i in devices:
                peak_alloc = torch.cuda.max_memory_allocated(i)
                peak_reserved = torch.cuda.max_memory_reserved(i)
                inc_alloc = max(0, peak_alloc - base_stats[i][0])
                inc_reserved = max(0, peak_reserved - base_stats[i][1])
                parts.append(
                    f'cuda:{i} inc_alloc={inc_alloc/1024**2:.1f}MB '
                    f'inc_reserved={inc_reserved/1024**2:.1f}MB '
                    f'peak_alloc={peak_alloc/1024**2:.1f}MB'
                )
            if hasattr(self, 'logger'):
                self.logger.info('[mem] ' + ' | '.join(parts))
            else:
                print('[mem] ' + ' | '.join(parts))
        return out

    HuggingFaceBaseModel.generate = wrapped


_wrap_generate_for_mem()

with read_base():
    from opencompass.datasets.needlebench_v2.origin import NeedleBenchOriginDataset
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_2m.needlebench_v2_single_2m import (  # noqa: E501
        needlebench_reader_cfg,
        needlebench_infer_cfg,
        needlebench_eval_cfg,
    )
    from opencompass.configs.summarizers.needlebench_v2_2m_summarizer import needlebench_v2_2m_summarizer as summarizer  # noqa: E501


datasets = [
    dict(
        abbr=f'Length{SMOKE_CONTEXT_LEN}Depth{SMOKE_DEPTH}_origin_en_2m_smoke',
        type=NeedleBenchOriginDataset,
        path='opencompass/needlebench',
        length=SMOKE_CONTEXT_LEN,
        depth=SMOKE_DEPTH,
        tokenizer_model='gpt-4',
        file_list=['PaulGrahamEssays.jsonl'],
        num_repeats_per_file=1,
        length_buffer=3000,
        language='English',
        needle_file_name='needles.jsonl',
        reader_cfg=needlebench_reader_cfg,
        infer_cfg=needlebench_infer_cfg,
        eval_cfg=needlebench_eval_cfg,
    )
]

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
        ),
        tokenizer_kwargs=dict(
            trust_remote_code=True,
            subfolder=TPTT_SUBFOLDER,
            padding_side='left',
        ),
        max_seq_len=2048000,
        max_out_len=256,
        batch_size=1,
        run_cfg=dict(num_gpus=2),
    )
]

work_dir = './outputs/needlebench_2m_titans_llama3_2_1b_smoke'
