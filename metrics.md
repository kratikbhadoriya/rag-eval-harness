# Metrics log

Every change that affects retrieval or answer quality gets a row.

| date | change | metric | before | after | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-14 | baseline | not yet measured | - | - | plain similarity search, no golden set yet |
| 2026-11-07 | baseline, k=5, SAMPLE=30, generator groq/openai/gpt-oss-120b, Judge gpt-4o-mini | faithfulness / context recall / refusals | not measured | 0.656 / 0.667 / 5 of 5 | thresholds set to baseline - 0.05 (0.606 / 0.617), refusal floor min(4, 5-1) = 4; Judge cost $0.035 |
| 2026-11-07 | generator groq/openai/gpt-oss-120b -> gpt-4.1-mini (Groq free tier daily token cap broke CI) | faithfulness / context recall / refusals | 0.656 / 0.667 / 5 of 5 | 0.793 / 0.667 / 5 of 5 | thresholds reset to baseline - 0.05 (0.743 / 0.617); answerable rows refused 6 -> 3 of 25; Judge cost $0.0385 |
| 2026-11-14 | retrieval k 5 -> 8 (#28), SAMPLE=30 | faithfulness / context recall / answerable refused | 0.793 / 0.667 / 3 of 25 | 0.875 / 0.773 / 2 of 25 | experiment only, Judge cost $0.0518 |
| 2026-11-14 | retrieval k 5 -> 10 (#28), SAMPLE=30 | faithfulness / context recall / answerable refused | 0.793 / 0.667 / 3 of 25 | 0.910 / 0.853 / 1 of 25 | experiment, adopted as default |
| 2026-11-14 | retrieval k 5 -> 15 (#28), SAMPLE=30 | faithfulness / context recall / answerable refused | 0.793 / 0.667 / 3 of 25 | 0.957 / 0.873 / 1 of 25 | experiment only; refusals no better than k=10 for 50% more context, not adopted |
| 2026-11-14 | default k=10 (#28), official run, SAMPLE=30 | faithfulness / context recall / refusals | 0.793 / 0.667 / 5 of 5 | 0.919 / 0.853 / 5 of 5 | thresholds reset to baseline - 0.05 (0.869 / 0.803), refusal floor unchanged; answerable refused 3 -> 1 of 25; chunk size and embedder not changed; Judge cost $0.0583 |
| 2026-11-14 | chunks 2000/400 -> 1000/200, k=10 (#28), SAMPLE=30 | faithfulness / context recall / answerable refused | 0.919 / 0.853 / 1 of 25 | 0.945 / 0.713 / 0 of 25 | experiment only, not adopted: context recall regressed 0.14 |
| 2026-11-14 | chunks 2000/400 -> 500/100, k=10 (#28), SAMPLE=30 | faithfulness / context recall / answerable refused | 0.919 / 0.853 / 1 of 25 | 0.852 / 0.650 / 1 of 25 | experiment only, not adopted: both metrics regressed |
| 2026-11-14 | embedder all-MiniLM-L6-v2 -> OpenAI text-embedding-3-small, k=10 (#28), SAMPLE=30 | faithfulness / context recall / answerable refused | 0.919 / 0.853 / 1 of 25 | 0.939 / 0.853 / 0 of 25 | experiment only, not adopted: gain is within run-to-run Judge noise (k=10 run twice gave 0.910 and 0.919) and it would make ingest and retrieval need an API key and network |
| 2026-11-15 | Retriever dense (#38), k=10, SAMPLE=30 | faithfulness / context recall / answerable refused | 0.919 / 0.853 / 1 of 25 (baseline before #38) | 0.926 / 0.853 / 2 of 25 | comparison run, Golden set version d62e7abc, no change adopted; Judge cost $0.0570 |
| 2026-11-15 | Retriever bm25 (#38), k=10, SAMPLE=30 | faithfulness / context recall / answerable refused | 0.926 / 0.853 / 2 of 25 (dense) | 0.896 / 0.927 / 1 of 25 | comparison run vs dense, Golden set version d62e7abc; context recall +0.073 over dense, +0.053 over runner-up hybrid, clears the 0.02 margin: bm25 wins; default change is #39; Judge cost $0.0577 |
| 2026-11-15 | Retriever hybrid (#38), k=10, SAMPLE=30 | faithfulness / context recall / answerable refused | 0.926 / 0.853 / 2 of 25 (dense) | 0.945 / 0.873 / 1 of 25 | comparison run vs dense, Golden set version d62e7abc; context recall +0.020 over dense, 0.053 below bm25; Judge cost $0.0566 |
| 2026-11-15 | default Retriever dense -> bm25 (#39), k=10, SAMPLE=30 | faithfulness / context recall / unanswerable refused | 0.919 / 0.853 / 5 of 5 (dense) | 0.896 / 0.927 / 5 of 5 (bm25) | thresholds reset to bm25 baseline - 0.05 (0.846 / 0.877), refusal floor unchanged (baseline_refusal_count 5); values from the #38 bm25 run, no new eval; faithfulness dropped 0.023 while context recall rose 0.073, both within the new thresholds |
| 2026-11-18 | bm25 k=10 -> k=5 (#41), SAMPLE=30 | faithfulness / context recall / answerable refused | 0.913 / 0.907 / 2 of 25 (bm25 k=10, re-run) | 0.813 / 0.813 / 5 of 25 | baseline for the rerank experiment, Golden set version d62e7abc, unanswerable refused 4 of 5; Context recall -0.093, Faithfulness -0.100 vs k=10; not adopted; Judge cost $0.0369 |
| 2026-11-18 | bm25 + cross-encoder rerank top-20 -> top-5 (#41), SAMPLE=30 | faithfulness / context recall / answerable refused | 0.913 / 0.907 / 2 of 25 (bm25 k=10); 0.813 / 0.813 / 5 of 25 (bm25 k=5) | 0.950 / 0.827 / 0 of 25 | experiment only, not adopted: Context recall -0.080 vs k=10 and +0.013 vs k=5, below the +0.02 over the better baseline required; Faithfulness +0.037 vs k=10 (+0.137 vs k=5); unanswerable refused 5 of 5; added rerank latency p50 1739 ms, p95 1889 ms, n=30 queries, Intel i5-8300H CPU (8 threads, no GPU), model cross-encoder/ms-marco-MiniLM-L-6-v2; Golden set version d62e7abc; Judge cost $0.0381 (k=10 re-run $0.0572); gate thresholds and defaults unchanged |

## Starting configuration

- Embedding model: Chroma default all-MiniLM-L6-v2 (local ONNX)
- Chunk size: 2000 characters
- Chunk overlap: 400 characters
- Retrieval: k=10 by default (was 5), bm25 Retriever by default (was dense; #39), score = raw BM25 score (dense: 1 - cosine distance)
