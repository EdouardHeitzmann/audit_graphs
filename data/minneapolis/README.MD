## Details/History.

Minneapolis has 2-seat STV for its City Board of Estimate and Taxation, and 3-seat STV for its Park Board.

There were political endorsements for some candidates. For example, in 2009
the Democratic-Farmer-Labor party endorsed Carol Becker, and the Minneapolis
City Republican Committee endorsed Michael Martens. These endorsements are not
encoded as partisan affiliations in the data.

Minneapolis has some unique STV rules -- in particular, they batch-eliminate all "mathematically impossible winners" whenever possible (which could theoretically lead to different outcomes than WIGM STV, if those impossible winners could otherwise stick around long enough to eliminate an otherwise promising candidate; but this is pretty unusual in practice).

## Data Source.

The 2009 - 2021 data comes from the [Harvard Dataverse](https://doi.org/10.7910/DVN/STVUET). The 2025 data comes directly from the [City of Minneapolis Website](https://vote.minneapolismn.gov/results-data/election-results/2025/bet/).

## Cleaning.

I left in ballots with preferences for "writein" because removing them would usually change the value of quota (or create empty ballots), which would be more confusing than leaving them in (note: the 2025 data distinguishes between "writein" and "UWI" -- I removed only "writein" in those cases, to keep the data consistent with the posted results).

I did remove "overvote" preferences from the profiles; the rule in Minneapolis (cf. [subsection c(2)](https://library.municode.com/mn/minneapolis/codes/code_of_ordinances?nodeId=COOR_TIT8.5EL_CH167MUELRUCO_167.70TAVOMUATEL) of the rules) is to skip such rankings (rather than truncate the ballot), so this is also what I did for the data herein contained.

I also removed many duplicate rankings from the ballots in these profiles -- it was very common to see the same candidate show up multiple times on the same ballot, in which case I just removed that candidate's occurences past the first (in accordance with with that same subsection of the rules).

## Oddities.

The 2009 Board of Estimate and Taxation data seems to be missing one vote for Carol Becker. For some reason, the official tabulation for that year calibrated winner tallies at `10697.556` (rather than the officially reported quota of `10686` votes) -- maybe because of a rounding rule I'm missing.

The 2025 Park Board [tabulation](https://tableau.minneapolismn.gov/views/2025ParkBoardAtLarge/2025ParkBoardAtLarge?%3Aembed=y&%3AisGuestRedirectFromVizportal=y) is hard to replicate past the seating of the first winner -- that tabulation shows Meg Forney receiving votes even after she first makes quota, presumably because Mary McKelvey was determined to be a "mathematically impossible winner" before Forney's transfer happened.

The 2013 Park Board election is singular -- John Erwin makes an exact quota in the data I have, although the official tabulation shows him having a non-trivial fractional surplus (`14,866` votes in my tabulation vs `14,866.2472` on the record). The tallies in the round after his election also don't perfectly match up with what has been posted online -- this could just be rounding.

The 2021 Park Board tabulation is similarly hard to replicate after the first seating -- presumably because of rounding protocols used.
