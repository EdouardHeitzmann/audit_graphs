# STV Audit Graphs

Research code for graph-based Risk-Limiting Audits of Single Transferable Vote
elections. Audit graphs enumerate the election paths an STV tabulation could
plausibly follow if fewer than a fixed Margin of Insecurity worth of votes were
misrecorded; auditing the graph's escape edges yields an RLA for the reported
outcome.

## Layout

- `src/election_graphs/` — abstract layered-graph constructor, shared
  datatypes, and profile utilities (maximum possible tallies, strong/weak
  candidate search).
- `src/wigm_graphs/` — plausible-graph constructors for WIGM STV, including
  batch elimination (seeded builds) and black-boxed seatings.
- `src/meek_graphs/` — sibling constructor for Meek STV.
- `src/test_processes/` — the audit machinery: edge-local compilers (test
  processes), vertex interpreters, global drivers, and the implicit
  ballot sampler.
- `src/experiments/` — end-to-end sample-size experiments backing the paper's
  results table.
- `src/margin_search/`, `src/symmetries.py`, `src/plotting.py` — supporting
  utilities: largest-coherent-margin search, candidate-symmetry quotients,
  and graph visualization.
- `notebooks/` — exploratory and results notebooks; `mismatch.ipynb`
  reproduces the paper's mismatch-based audits.
- `tests/` — pytest suite; run with `uv run pytest tests/`.

Profiles are loaded with [votekit](https://github.com/mggg/VoteKit). See
`CLAUDE.md` for a map from the paper's conceptual objects to the code.
