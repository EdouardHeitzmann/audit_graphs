# Election Data

Votekit-format ballot profiles for the elections in the writeup's results
table, organized by source jurisdiction. Each subdirectory carries the license
and attribution for its data.

These files are a small excerpt of a larger election database maintained at
[EdouardHeitzmann/election_data](https://github.com/EdouardHeitzmann/election_data.git),
which documents the provenance, cleaning, and format conversions in detail.

Format (Scottish election csv): the first row is `candidates,seats`; each
following row is a ballot weight followed by a ranking; the file ends with the
candidate names and the ward name.

Note on write-ins: the raw Minneapolis and Albany profiles include a write-in
pseudo-candidate (`writein` in Minneapolis 2021, `UWI` in Minneapolis 2025,
`write-in` in Albany). The results table reports these elections with the
write-in removed via
`src.election_graphs.numpy_profile.remove_and_condense_numpy_profile`.
Minneapolis keeps the emptied ballots, because its official thresholds are
computed from the full ballot weight (see
`tests/replication/test_minneapolis_preprocessing.py`); Albany drops them,
matching Alameda County's pre-exhaustion of write-ins before round one.
