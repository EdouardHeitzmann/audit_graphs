# Research Staging.

This is a late-stage research project that is near completion. 
The current writeup is finished and reflects what will be published out of this project.
The remaining work is to clean up and refactor this codebase to make it ready for publication.

# Conceptual Objects.

## Audit Graphs.

The main conceptual objects in play here are audit graphs. 
These are oriented, directed, layered, rooted graphs.
Their vertices represent possible stages of an STV election tabulation: they contain information about which "hopeful" candidates remain, which were seated, and the seating *edges* at which each winner was seated (the `seated_at` tuple; each seating edge carries the post-transfer ballot weights).
During construction, vertices assign tallies to each of their hopeful candidates by treating the election sequence preceding them as the ground truth.
Edges represent possible decisions the tabulation algorithm could make: usually eliminate a candidate or seat some candidate(s).
There are three main procedures we use to construct a graph: plausible depth-first search construction, batch elimination, and black-boxed seating.

### Plausible DFS.

The terms "Least Auditable Margin" (`LAM`) and "Margin of Insecurity" (`MoI`) are interchangeable (the former is the legacy version of the latter).
They both describe an integer $M$ that is fixed ahead of graph construction.
Starting from the root vertex, the DFS approach follows any edge that is "plausible," in the sense that it could have been the edge chosen by the true tabulation algorithm if a margin of less than $M$ votes flipped between two tallies in the edge's base vertex.
These are contained in `src/wigm_graphs/graph_wigm.py`.
They are formally defined in Sections 2.1 and 2.2 of the writeup.

### Batch Elimination.

For large elections, full DFS construction is computationally intractable.
Batch elimination is a solution to this problem which rigorously justifies the simultaneous elimination of many "weak candidates."
The mechanics of this are descibed in Section 2.3 of the writeup.
Their code is contained in the `seeded_build` method of `SeededWIGMGraphConstructor` in `src/wigm_graphs/seeded.py`, a subclass of the plain DFS constructor.

### Black-Boxed Seating.

When a strong candidate `w` would make quota during a batch elimination, we use a black-boxed seating to seat them in a way that is chronology-agnostic.
This procedure is not described anywhere in the writeup.
The code for it is in `src/wigm_graphs/black_box.py` (`BlackBoxWIGMGraphConstructor`).
Like batch elimination, this procedure uses a "base vertex," which is the last vertex resulting from the seating of the very strong candidates.
A ballot is an "uncertain vote" for a candidate `l` when its first-place vote in the base vertex is *not* `w`, but `w` appears on the ballot before `l` (its current top choice post-seeding), with no strong candidate in between.
We cannot tell whether such a ballot transferred through `w` (reaching `l` deflated by `w`'s transfer value) or skipped an already-seated `w` (reaching `l` at full base-vertex weight); these votes are what the candidate-to-mentions margins for the weak candidates must account for, and post-seeding the uncertainty only matters through worst-case bounds on `w`'s transfer value.
Accordingly, vertices in black-box graphs treat tallies differently: they hold *interval* tallies (per-scenario certain tallies from the transfer-value bounds, plus a separately tracked uncertain mass), and `vertex.tallies` exposes only the lower bound.
When deciding whether an edge is plausible, we assign the uncertain mass in the way that makes the edge look as plausible as possible, to remain conservative.

### Meek Graphs.

`src/meek_graphs/graph_meek.py` contains a sibling constructor for Meek STV: it recalibrates keep factors per vertex, so its state keys are chronology-free, and it adds a `reverse_build` mode that WIGM lacks.
It supports neither simultaneous elections nor black-boxed seating, and the audit drivers do not accept it.

## Test Processes.

The main application of audit graphs is to create RLAs of STV elections.

### Drivers.

These take in an audit graph and create the edge-local compilers needed to reject each of the local nulls.
They also handle the noising of ballots for simulated audits, and they broadcast ballot comparisons to their compilers.
Examples of drivers include `GlobalAuditDriver` and `GlobalAuditDriverV2` in `src/test_processes/driver.py`, and `DeltaMethodAuditDriver` in `src/test_processes/delta_method.py`.
No driver currently accepts black-box seeded graphs.

### Compilers.

Compilers are the codebase's name for test processes. 
Each compiler corresponds to a simple edge-local null.
It is initialized knowing a critical margin (either candidate-to-candidate or candidate-to-quota, or sometimes candidate-to-mentions), and it is equipped with a method to use the ballot comparisons that the driver broadcasts to it to update.
In all cases except the `DeltaMethodCompiler`, this updating method just uses the ballot comparison to update a martingale capital.
Most of the other compilers are contained in `src/test_processes/cobra.py`. 

### Vertex Interpreters.

In some cases, the driver broadcasts comparisons to vertex-local interpreters, which amortizes some of the pre-processing of the comparison before broadcasting it to each of the compilers for its escape edges.

## Supporting Utilities (local-only).

`src/margin_search/` (searches for the largest coherent MoI), `src/symmetries.py` (candidate-equivalence quotients of graphs), and the optimizer modules `stv_partial_optimizer.py` / `noise_filtered_linearizer.py` (worst-case discrepancy bounds in the vertex-local `t_{S,x}` coordinates, used by the V2 compilers and the delta method) support the objects above.
None of these appear in the writeup; they stay in the local copy but will be pruned from the published repository.
