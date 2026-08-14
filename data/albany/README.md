## Details/History.

Albany, California elects its City Council and Board of Education at large
using proportional STV, first used in November 2022. Both bodies have five
members with staggered terms, so each cycle fills two or three seats: in 2022,
2 Council seats and 3 Board of Education seats; in 2024, 3 Council seats and 2
Board of Education seats. Tabulation is run by the Alameda County Registrar of
Voters.

Albany's STV rules are given in
[section 7 of their municipal code](https://ecode360.com/36944383).

## Data Source.

The cast vote records come from the [Harvard Dataverse](https://doi.org/10.7910/DVN/STVUET),
the same FairVote dataset as the Cambridge data.

I treated the Alameda County round-by-round tabulation reports (e.g.
[2022 City Council](https://www.alamedacountyca.gov/rovresults/rcv/248/rcvresults.htm?race=Albany%2F001-CityCouncil))
as the source of truth when checking these profiles.

## Cleaning.

Write-in candidates are kept in the profiles, appearing as the single
candidate `write-in`.

Under section 7-1.5, a ballot with an overvote counts only up to the ranking
position where the overvote occurs, so I truncated each such ballot at that
position and discarded all later rankings.

## Oddities.

The official tabulation reports pre-exhaust write-in candidates before round
one, so their first-round first-preference totals differ slightly from the
totals obtained by tabulating these profiles directly.

Albany's first STV election in 2022 used an unusual form of Droop quota, not
sanctioned by their rules, which was recalculated each round as a function of
non-exhausted ballot weight. They switched away from this in 2024. That original form
of quota is achievable by the `AlbanySTV` class in `votekit.elections`.
