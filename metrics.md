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


## Starting configuration

- Embedding model: Chroma default all-MiniLM-L6-v2 (local ONNX)
- Chunk size: 2000 characters
- Chunk overlap: 400 characters
- Retrieval: k=10 by default (was 5), plain nearest-neighbour, score = 1 - cosine distance
