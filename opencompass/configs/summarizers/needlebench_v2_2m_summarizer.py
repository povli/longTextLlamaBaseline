from opencompass.configs.summarizers.needlebench import create_summarizer

depths_list_10 = [i for i in range(0, 101, 10)]
context_lengths_2m = [32000, 64000, 128000, 512000, 1000000, 2000000]

needlebench_v2_2m_summarizer = create_summarizer(
    context_lengths_2m, depths_list_10, '2m', mean=True)
