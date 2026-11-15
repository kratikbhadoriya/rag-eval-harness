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

**Retriever**:
A way of ranking chunks for a query: dense (by meaning), bm25 (by shared words) or hybrid (the two combined).
_Avoid_: Search, ranker, index

**Reciprocal rank fusion**:
Combining several retrievers' rankings by giving each chunk a share per ranking that shrinks with its rank, then ordering chunks by their total.
_Avoid_: Blending, ensemble, merge

**Score**:
How well a chunk matches a query under the retriever that ranked it, where higher means a better match. Scores from different retrievers are not comparable.
_Avoid_: Distance, relevance, confidence

**Golden set version**:
A short fingerprint of the Golden set's contents, recorded with every measurement so runs are only compared when it matches.
_Avoid_: Revision, release

**Golden set**:
The hand-written questions, each paired with a reference answer and the source pages that support it, used to measure the harness. Frozen once written.
_Avoid_: Test set, benchmark, eval data

**Frozen corpus**:
The Corpus as checksummed in `data/corpus/SHA256SUMS`; it never changes, so scores stay comparable across weeks.
_Avoid_: Snapshot, fixture

**Unanswerable question**:
A golden-set question the Corpus cannot answer, where the correct response is to say so.
_Avoid_: Negative, trick question, out-of-scope

**Judge**:
The LLM, separate from the one that writes answers, that scores each answer for faithfulness and context recall.
_Avoid_: Grader, evaluator, scorer

**Faithfulness**:
How much of an answer's content is supported by the contexts it was given, from 0 to 1.
_Avoid_: Accuracy, correctness, groundedness

**Context recall**:
How much of the reference answer is covered by the contexts that were retrieved, from 0 to 1.
_Avoid_: Retrieval accuracy, hit rate

**Refusal**:
An answer that says the Corpus does not contain what was asked, using the fixed refusal phrase from the prompt.
_Avoid_: Abstention, no-answer

**Quality gate**:
The CI check that fails a change when faithfulness or context recall on answerable questions falls below its threshold.
_Avoid_: Test, benchmark, check

**Refusal gate**:
The CI check, run alongside the quality gate, that fails a change when too few Unanswerable questions get a Refusal.
_Avoid_: Safety gate
