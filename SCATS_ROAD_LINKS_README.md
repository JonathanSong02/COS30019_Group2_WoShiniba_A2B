# SCATS Road Links — PROVISIONAL teaching/demo graph (15 November 2026)

**Do not describe this as a verified road graph or measured driving distances.**

Files:
- `scats_road_links.csv` — exactly the three columns required by Group 2 `main.py`: `from_site,to_site,distance_km`.
- `scats_road_links_review.csv` — source road names, provisional status and validation notes for each undirected pair.

## Basis and limits

The connections are a preliminary corridor network built from the 40 site names and coordinates in Group 2's `data/processed/scats_sites.csv`, read alongside the lecturer's **IBM data_Bor.pdf** (VicRoads/ESRI Boroondara map, July 2009). Consecutive SCATS sites with matching named roads were linked. Each pair is written in both directions as an **assumption** that must be checked against turn restrictions and one-way streets. The provided map labels many additional SCATS sites not in this 40-site training subset; links may therefore skip unmodelled controlled intersections.

`distance_km` is calculated using the geographic great-circle / Haversine *straight-line* distance between coordinates and rounded to three decimals. This is a **lower bound, not a verified road distance**. The original documents do not supply road centreline polylines or measured route segment lengths. **Replace these values with actual road distances before using the network for final travel-time claims.** Do not claim the approximate graph is an externally verified SCATS road topology.

## Integration

Copy `scats_road_links.csv` into: `data/processed/scats_road_links.csv`. The CSV loader accepts these columns unchanged. There are 48 undirected candidate pairs, 96 directed rows, and the largest component contains 38 of the 40 sites. The remaining sites are isolated rather than connected through invented roads.

The existing `main.py route` command also requires saved ML checkpoints for each site where traffic is used. Group 2 currently has GRU checkpoints for **site 2000 only**. Loading this network does **not** by itself make real-data routing work for all nodes. Also follow the course's 1,500 vehicle/hour capacity limit; `steady_4x` is an assumption, not an observed hour.

## Verification work to complete

1. Compare every row in `scats_road_links_review.csv` against a routable OpenStreetMap/VicRoads road graph or actual road-centreline GIS data and remove false links.
2. Confirm which direction(s) traffic is permitted for every segment; exclude roads with turn restrictions or one-way constraints.
3. Obtain measured **road distances** in km (not Haversine) using the road geometry.
4. Identify links skipping unsampled SCATS sites and document whether the intended graph should include those intermediate nodes.
5. Train/load prediction models for the remaining required SCATS sites before using `main.py route`.

## Evidence

Sources: lecturer-supplied `IBM data_Bor.pdf` and Group 2's processed `scats_sites.csv`; no live road-network or routing service was accessible for independent validation.
