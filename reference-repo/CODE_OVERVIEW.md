# What each file does

`keywordextractor.py` - entry point, dispatches to one of three commands: `download-example-dataset`, `annotate`, `evaluate`.

`modules/commands.py` - the three commands themselves:
- `download_example_dataset()`: downloads the full D3 papers dump, filters it down to the 2500 papers used in the paper, saves `metadata.json` (titles, abstracts, CSO subjects) and `targets.json` (the 19 topics)
- `annotate()`: runs the 3-turn prompt on every document via the local LLM, saves results to `annotated_metadata.json`
- `evaluate()`: reads the annotated file and prints the success/hallucination stats

`modules/prompt_engineering.py` - builds the exact 3-turn prompts (Figure 2 in the paper) and parses the LLM's raw answer into matched topics.

`modules/data_processing.py` - the matching logic: lowercases and trims an answer, checks it against the 19 topics for an exact match. This is the part our `src/matching.py` improves on.

`modules/data_loading.py` / `modules/data_fetching.py` - reading/writing the JSON dataset files and unzipping the downloaded D3 archive.

`modules/evaluation.py` - computes the average/min/max stats used by the `evaluate` command.

`modules/configuration.py` - loads `config/config-defaults.ini`, overridden by an optional `config/config.ini`.

`classes/gpt4all_model.py` - wraps GPT4All, runs the 3 prompts as one real chat session (each prompt gets a reply, replies stay in context for the next prompt).

`reproducibility/dataset_D3_ID_list.json` - the exact 2500 D3 paper IDs used in the paper, so `download-example-dataset` can recreate the same dataset instead of a random sample.
