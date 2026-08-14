## Details/History.

This directory contains New South Wales local government STV elections from
2012 through 2024. Files are grouped by number of candidates.

## Data Source.

The `.stv` source profiles came from Andrew Conway's ConcreteSTV election
database and were derived from NSW Electoral Commission data.

## Cleaning.

Where ATL votes were present, they were expanded to BTL candidate rankings to
make the profiles compatible with VoteKit's Scottish CSV format. Missing party
affiliations are recorded as `None`.
