# What's in this folder

`metadata.json` - the actual dataset we use. 2500 papers, each with D3 ID, title, abstract, and its real subjects (topics) from the CSO classifier. This is the ground truth for every experiment.

`targets.json` - the list of 19 valid topics (the controlled vocabulary).

`used_D3_ID_list.json` - just the 2500 D3 IDs that ended up in metadata.json, generated as a side effect, not actively used by anything.

`d3_papers.jsonl` (15.7GB) and `d3_papers.jsonl.gz` (3.7GB) - the full D3 papers dump these 2500 were filtered out of. Only needed once, during the download step. Safe to delete to free up disk space, metadata.json already has everything we need.
