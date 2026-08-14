# STV Audit Graphs

Research code for graph-based Risk-Limiting Audits of Single Transferable Vote
elections, accompanying the writeup in `writeup/`. Audit graphs enumerate the
election paths an STV tabulation could plausibly follow if fewer than a fixed
Margin of Insecurity worth of votes were misrecorded; auditing the graph's
escape edges yields an RLA for the reported outcome.

## Start here: the demo notebooks

Two notebooks walk through the whole pipeline step by step:

- [`notebooks/mismatch.ipynb`](notebooks/mismatch.ipynb) — the DFS path:
  build a plausible audit graph by depth-first search for Portland District 1
  2024 (the writeup's Figure 1 election), plot it, then run both audit
  drivers — the mismatch audit demonstrating its failure mode at 2% noise,
  the Delta-method audit certifying at 711 ballots.
- [`notebooks/victoria_v2.ipynb`](notebooks/victoria_v2.ipynb) — the
  large-election path: batch-elimination (seeded) construction for the 2025
  Victorian Senate election (65 candidates, 4.1M ballots), a plot of the
  abridged graph, a certifying Delta-method audit at 0.5% of ballots, and a
  bounded mismatch-driver demonstration.

## Reproducing the results table

Set up an environment and run the tests:

```sh
uv venv && uv pip install numpy pandas scipy sympy matplotlib pytest
.venv/bin/python -m pytest tests/
```

Re-create the full results table of Section 3.4 — every populated row, at its
maximal coherent Margin of Insecurity, with the graph construction method the
table's footnotes declare (2% noised CVRs, risk level 5%):

```sh
.venv/bin/python -m src.replication.reproduce_full_table_asns
```

Row selection and the audit parameters are configurable — `--rows 1 3` runs a
subset (rows are 1-indexed in table order), and `--help` lists the rest. Rows
whose Mismatch ASN the table marks X skip the mismatch audit by default, since
those audits run to their sample cap. The Australian Senate rows are by far
the most expensive; everything through Minneapolis finishes in hours on a
desktop.

The Mismatch ASN averages 10 simulated audits (all must certify below half the
ballots); the Delta ASN is the smallest sample size certifying at least 9 of
10 seeded trials. Both are seed-dependent statistics, so reproduced values can
differ slightly from the table. `src/replication/reproduce_table_asns.py` is a
smaller variant covering the table's first six rows.

Individual elections can be studied directly with the statistics gatherer,
which exposes the same protocol for one profile at an arbitrary margin:

```sh
.venv/bin/python -m src.replication.driver_statistics \
    data/scot-elex/eilean_siar_2022_ward5.csv --enforced-moi 141
```

## Layout

- `src/election_graphs/` — abstract layered-graph constructor, shared
  datatypes, and profile utilities (numpy-profile loading and condensing,
  maximum possible tallies, strong/weak candidate search).
- `src/wigm_graphs/` — plausible-graph constructors for WIGM STV: the plain
  DFS constructor and the batch-elimination (seeded) subclass.
- `src/test_processes/` — the audit machinery: edge-local compilers (test
  processes), vertex interpreters, the mismatch and Delta-method drivers, the
  shared escape-margin selection, the symbolic margin equations backing the
  Delta method, and the implicit ballot sampler.
- `src/margin_search/` — heap-based search for the largest Margin of
  Insecurity admitting a coherent plausible graph.
- `src/replication/` — the end-to-end scripts described above.
- `src/plotting.py` — graph visualization.
- `data/` — votekit-format ballot profiles for the elections in the results
  table, organized by source jurisdiction with their licenses. See
  `data/README.md` for provenance and the write-in preprocessing protocols.
- `writeup/` — the paper's LaTeX sources; the definitions there are the
  source of truth for the code.
- `tests/` — pytest suite.

Profiles are loaded with the vendored numpy-profile reader in
`src/election_graphs/numpy_profile.py` (Scottish election csv format, as
produced by [votekit](https://github.com/mggg/VoteKit)). See `CLAUDE.md` for a
map from the paper's conceptual objects to the code.
