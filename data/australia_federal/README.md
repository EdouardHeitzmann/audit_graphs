## Details/History.

Australian Senate elections have used STV since at least 1949; the Australian
Electoral Commission has published full CVRs for these elections since 2016.
The Northern Territory and Australian Capital Territory fill two vacancies.
The states usually fill six vacancies, except during a double dissolution,
when they fill twelve. The 2016 election is the only double dissolution in
this repository.

## Data Source.

The data here was downloaded directly from the AEC; Conway's `.stv` source
files were not used for this conversion.

## Cleaning.

As mentioned in the root README, I had to convert ATL votes to BTL votes to make this data compatible with Scottish format. The data was otherwise already clean; there was no need for me to modify it in any additional way.

## File Storage.

Files smaller than 50 MB are stored directly as CSV files. Larger files are
stored as individual ZIP archives in `zipped/` to stay below GitHub's file-size
limit. Each archive contains one CSV with the matching base name.

To extract every archived CSV and place it alongside the uncompressed CSVs,
run the following commands from the repository root:

```bash
cd votekit/australia_federal
for archive in zipped/*.zip; do
    unzip -jo "$archive" -d .
done
```

The `-j` option discards directory paths stored inside an archive, while `-o`
replaces an existing local copy without prompting.
