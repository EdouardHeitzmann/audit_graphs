# STV Audit Graphs

Research code for graph-based Risk-Limiting Audits of Single Transferable Vote
elections. Audit graphs enumerate the election paths an STV tabulation could
plausibly follow if fewer than a fixed Margin of Insecurity worth of votes were
misrecorded; auditing the graph's escape edges yields an RLA for the reported
outcome.

## Layout

- `src/election_graphs/` — abstract layered-graph constructor, shared
  datatypes, and profile utilities (numpy-profile loading and condensing,
  maximum possible tallies, strong/weak candidate search).
- `src/wigm_graphs/` — plausible-graph constructors for WIGM STV, including
  batch elimination (seeded builds).
- `src/test_processes/` — the audit machinery: edge-local compilers (test
  processes), vertex interpreters, the mismatch and Delta-method drivers, the
  symbolic margin equations backing the Delta method, and the implicit ballot
  sampler.
- `src/margin_search/` — standalone utility searching for the largest Margin
  of Insecurity admitting a coherent plausible graph.
- `src/replication/` — end-to-end scripts re-creating the paper's results
  table.
- `src/plotting.py` — graph visualization for the writeup's figures.
- `data/` — votekit-format ballot profiles for the elections in the results
  table, organized by source jurisdiction with their licenses. See
  `data/README.md` for provenance.
- `notebooks/` — exploratory and results notebooks.
- `tests/` — pytest suite.

## Reproducing the results

Set up an environment and run the tests:

```sh
uv venv && uv pip install numpy pandas scipy sympy matplotlib pytest
.venv/bin/python -m pytest tests/
```

Re-create the first six rows of the results table (Section 3.4 of the
writeup) — graph construction at the table's Margin of Insecurity, 2% noised
CVRs, risk level 5%:

```sh
.venv/bin/python -m src.replication.reproduce_table_asns
```

Pass `--rows 1 3` to run a subset, and see `--help` for the audit parameters.
The whole table (all 22 populated rows, with each row's graph construction
method, skipping the mismatch audits the table marks X) is re-created by the
long-running

```sh
.venv/bin/python -m src.replication.reproduce_full_table_asns
```

Individual elections can also be studied directly with the statistics
gatherer, e.g.:

```sh
.venv/bin/python -m src.replication.driver_statistics \
    data/scot-elex/eilean_siar_2022_ward5.csv --enforced-moi 141
```

The Mismatch ASN averages 10 simulated audits (all must certify below half the
ballots); the Delta ASN is the smallest sample size certifying at least 9 of
10 seeded trials. Both are seed-dependent statistics, so reproduced values can
differ slightly from the table.

Profiles are loaded with the vendored numpy-profile reader in
`src/election_graphs/numpy_profile.py` (Scottish election csv format, as
produced by [votekit](https://github.com/mggg/VoteKit)). See `CLAUDE.md` for a
map from the paper's conceptual objects to the code.
