# Gate only on Faithfulness and Context recall

The Quality gate checks Faithfulness and Context recall on answerable questions, and the Refusal gate checks Unanswerable questions. Answer relevancy is deliberately excluded: it is noisy at the small size of the Golden set, so gating on it would fail honest changes and teach contributors to ignore the gate.

## Considered Options

- Gate on all three Judge metrics, including answer relevancy. Rejected because a failure from noise alone is indistinguishable from a real regression.
- Gate on Faithfulness and Context recall only. Chosen: each tracks a distinct, stable failure (unsupported answers, missing evidence).

## Consequences

- A regression that only lowers answer relevancy passes the gate unnoticed.
- Adding answer relevancy later requires evidence that its noise is small relative to the threshold margin, and a new ADR superseding this one.
