# Metrics log

Every change that affects retrieval or answer quality gets a row.

| date | change | metric | before | after | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-14 | baseline | not yet measured | - | - | plain similarity search, no golden set yet |

## Starting configuration

- Embedding model: Chroma default all-MiniLM-L6-v2 (local ONNX)
- Chunk size: 2000 characters
- Chunk overlap: 400 characters
- Retrieval: plain nearest-neighbour, score = 1 - cosine distance
