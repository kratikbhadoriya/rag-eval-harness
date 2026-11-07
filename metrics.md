# Metrics log

Every change that affects retrieval or answer quality gets a row.

| date | change | metric | before | after | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-14 | baseline | not yet measured | - | - | plain similarity search, no golden set yet |
| 2026-11-07 | baseline, k=5, SAMPLE=30, generator groq/openai/gpt-oss-120b, Judge gpt-4o-mini | faithfulness / context recall / refusals | not measured | 0.656 / 0.667 / 5 of 5 | thresholds set to baseline - 0.05 (0.606 / 0.617), refusal floor min(4, 5-1) = 4; Judge cost $0.035 |
| 2026-11-07 | generator groq/openai/gpt-oss-120b -> gpt-4.1-mini (Groq free tier daily token cap broke CI) | faithfulness / context recall / refusals | 0.656 / 0.667 / 5 of 5 | 0.793 / 0.667 / 5 of 5 | thresholds reset to baseline - 0.05 (0.743 / 0.617); answerable rows refused 6 -> 3 of 25; Judge cost $0.0385 |

## Starting configuration

- Embedding model: Chroma default all-MiniLM-L6-v2 (local ONNX)
- Chunk size: 2000 characters
- Chunk overlap: 400 characters
- Retrieval: plain nearest-neighbour, score = 1 - cosine distance
