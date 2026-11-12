# SCATS October 2006 - data quality report

- Raw rows (approach-days): 4192
- SCATS sites: 40; distinct approaches: 140
- Date range: 2006-10-01 to 2006-10-31 (31 days, 96 x 15-min intervals per day)
- Duplicate (site, approach, day) rows: 0
- Rows with zero coordinates: 31 (ignored for site coordinates)
- Negative or missing raw counts: 0
- Max 15-min count at one approach: 695
- Max 15-min site total: 1409
- Site-level grid: 119040 rows, of which 5184 missing (4.4%)

## Missing 15-min steps per split

| split | rows | missing |
|---|---|---|
| train | 84480 | 960 |
| val | 15360 | 2304 |
| test | 19200 | 1920 |

## Sites with missing data

| site | missing 15-min steps | missing days |
|---|---|---|
| 970 | 96 | 1 |
| 2000 | 288 | 3 |
| 2820 | 96 | 1 |
| 3002 | 480 | 5 |
| 3127 | 480 | 5 |
| 3682 | 192 | 2 |
| 3685 | 960 | 10 |
| 4057 | 192 | 2 |
| 4262 | 480 | 5 |
| 4263 | 480 | 5 |
| 4264 | 864 | 9 |
| 4272 | 96 | 1 |
| 4812 | 480 | 5 |

## Ignored non-core approaches

- 3001: CHURCH_ST SW of BARKERS_RD #6

Sites table: 40 sites with coordinates.
