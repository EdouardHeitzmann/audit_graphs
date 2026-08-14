# Publication Repository.

This repository accompanies the writeup in `writeup/` and contains exactly the
material presented there: plausible audit-graph construction for WIGM STV,
batch elimination, and the mismatch and Delta-method audit drivers, together
with the data and scripts needed to re-create the results table of Section 3.4.

# Conceptual Objects.

## Audit Graphs.

The main conceptual objects in play here are audit graphs.
These are oriented, directed, layered, rooted graphs.
Their vertices represent possible stages of an STV election tabulation: they contain information about which "hopeful" candidates remain, which were seated, and the seating *edges* at which each winner was seated (the `seated_at` tuple; each seating edge carries the post-transfer ballot weights).
During construction, vertices assign tallies to each of their hopeful candidates by treating the election sequence preceding them as the ground truth.
Edges represent possible decisions the tabulation algorithm could make: usually eliminate a candidate or seat some candidate(s).

### Plausible DFS.

The "Margin of Insecurity" (`MoI`) is an integer $M$ that is fixed ahead of graph construction.
Starting from the root vertex, the DFS approach follows any edge that is "plausible," in the sense that it could have been the edge chosen by the true tabulation algorithm if a margin of less than $M$ votes flipped between two tallies in the edge's base vertex.
These are contained in `src/wigm_graphs/graph_wigm.py`.
They are formally defined in Sections 2.1 and 2.2 of the writeup.

### Batch Elimination.

For large elections, full DFS construction is computationally intractable.
Batch elimination is a solution to this problem which rigorously justifies the simultaneous elimination of many "weak candidates."
The mechanics of this are described in Section 2.3 of the writeup.
Their code is contained in the `seeded_build` method of `SeededWIGMGraphConstructor` in `src/wigm_graphs/seeded.py`, a subclass of the plain DFS constructor.

## Test Processes.

The main application of audit graphs is to create RLAs of STV elections.

### Drivers.

These take in an audit graph and create the edge-local compilers needed to reject each of the local nulls.
They also handle the noising of ballots for simulated audits, and they broadcast ballot comparisons to their compilers.
`GlobalAuditDriver` in `src/test_processes/driver.py` runs the mismatch-based audits of Section 3.2, and `DeltaMethodAuditDriver` in `src/test_processes/delta_method.py` runs the Delta-method audits of Section 3.3.

### Compilers.

Compilers are the codebase's name for test processes.
Each compiler corresponds to a simple edge-local null.
It is initialized knowing a critical margin (either candidate-to-candidate or candidate-to-quota, or sometimes candidate-to-mentions), and it is equipped with a method to use the ballot comparisons that the driver broadcasts to it to update.
The COBRA martingale compilers live in `src/test_processes/cobra.py`: `CobraCompiler` tests an escape edge's critical margin directly, while the noise-filter compilers (subclasses of `CobraNoiseFilterBase`) test the local noise allowance of an escape edge in its reduced parameter space.
The `DeltaMethodCompiler` in `src/test_processes/delta_method.py` instead builds a Wald confidence interval for the critical margin at a fixed sample size, using the symbolic margin machinery in `src/test_processes/symbolic_equations.py`.

### Vertex Interpreters.

In some cases, the driver broadcasts comparisons to vertex-local interpreters (`src/test_processes/interpreter.py`), which amortizes some of the pre-processing of the comparison before broadcasting it to each of the compilers for its escape edges.

## Data and Experiments.

`data/` holds the votekit-format ballot profiles for the elections in the results table (Section 3.4), organized by source jurisdiction with their licenses.
`src/replication/driver_statistics.py` is the end-to-end statistics gatherer: it constructs a graph at an enforced MoI and measures the Mismatch and Delta ASNs following the protocol of Section 3.4.
`src/replication/reproduce_table_asns.py` hard-codes the table's MoI values for the first six rows and re-creates their ASNs.
