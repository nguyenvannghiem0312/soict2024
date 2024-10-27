import vocabtrimmer

trimmer = vocabtrimmer.VocabTrimmer("jinaai/jina-reranker-v2-base-multilingual", double_embedding=False)
trimmer.trim_vocab(
        path_to_save="jinaai/jina-reranker-v2-base-multilingual",
        language="vi",
        dataset='vocabtrimmer/mc4_validation',
        dataset_column='text',
        dataset_split='validation',
        target_vocab_size=60000,
        min_frequency=2,
        chunk=500,
        cache_file_vocab=None,
        cache_file_frequency=None,
        overwrite=False
    )