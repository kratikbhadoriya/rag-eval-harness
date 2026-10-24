# RAG Eval Harness

A small harness that ingests PDFs, retrieves passages for a query, and measures retrieval and answer quality against a golden set.

## Language

**Corpus**:
The set of PDF files under `data/corpus/` that the harness ingests and answers questions from.
_Avoid_: Dataset, documents, knowledge base

**Source**:
The PDF file a chunk came from, identified by its file name.
_Avoid_: Document, file path, origin

**Chunk**:
A contiguous span of text from a single page of a source, the unit that is embedded, stored and retrieved.
_Avoid_: Passage, segment, node, document

**Collection**:
A named group of chunks that are ingested and searched together.
_Avoid_: Index, database, store

**Score**:
How similar a chunk is to a query, where higher means more similar.
_Avoid_: Distance, relevance, confidence

**Golden set**:
The hand-written questions, each paired with a reference answer and the source pages that support it, used to measure the harness. Frozen once written.
_Avoid_: Test set, benchmark, eval data

**Frozen corpus**:
The Corpus as checksummed in `data/corpus/SHA256SUMS`; it never changes, so scores stay comparable across weeks.
_Avoid_: Snapshot, fixture

**Unanswerable question**:
A golden-set question the Corpus cannot answer, where the correct response is to say so.
_Avoid_: Negative, trick question, out-of-scope
