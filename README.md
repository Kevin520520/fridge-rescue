# Fridge Rescue

Fridge Rescue is a bilingual recipe recommender for ingredients already available at home.
It combines a curated local collection of 2,000 recipes, deterministic retrieval, a
LangGraph workflow, and an optional DeepSeek or OpenAI model.

This repository contains the runnable application and the reproducible corpus-building
pipeline. Development tests, experiment outputs, the 170 MB raw source, and audit archives
are intentionally excluded.

## What is included

- **Offline retrieval preview:** works without an API key.
- **Agentic AI workflow:** retrieves recipe candidates, lets the model select recipe IDs,
  validates the result, and performs at most one repair.
- **Single-prompt comparison:** calls the same configured model without the local corpus or
  workflow.
- **Curated corpus:** 2,000 Food.com recipes stored locally in Parquet format.
- **Reproducible data pipeline:** optional scripts download the fixed Kaggle source,
  inspect its schema, clean and deduplicate records, and rebuild the 2,000-recipe corpus.
- **Chinese ingredient aliases:** a reviewed set of common Chinese ingredient names is
  normalized to the English corpus vocabulary.

## Requirements

- Python 3.11, 3.12, or 3.13
- Internet access for installing packages, using a live AI mode, or optionally rebuilding data
- A personal DeepSeek or OpenAI API key for live AI modes

The processed 2,000-recipe corpus is already included. Normal users do **not** need to
download or process the Kaggle data again.

## Get the project

After accepting the private GitHub repository invitation, either download the repository
as a ZIP from **Code → Download ZIP**, or clone it:

```powershell
git clone https://github.com/Kevin520520/fridge-rescue.git
cd fridge-rescue
```

## Run locally on Windows

Open PowerShell in the extracted or cloned project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If Python 3.13 is unavailable, replace `-3.13` with `-3.12` or `-3.11`.

Streamlit will show a local address such as `http://localhost:8501`. Open it in a browser.
Without an API key, the application automatically offers the offline retrieval preview.
Press `Ctrl+C` in PowerShell to stop the application.

## Enable the live AI modes

Create a private local configuration file:

```powershell
Copy-Item .env.example .env
```

Open `.env`, replace `replace_with_your_own_key` with your own DeepSeek key, save the file,
and restart Streamlit. Never upload `.env` or share an API key in screenshots or chat.

The default configuration is:

```text
LLM_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

OpenAI is an optional alternative; instructions are provided in `.env.example`.

## Run on macOS or Linux

```bash
git clone https://github.com/Kevin520520/fridge-rescue.git
cd fridge-rescue
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m streamlit run app.py
```

## Optional: rebuild the 2,000-recipe corpus

This section is for reproducing the data preparation process, not for normal application
use. It downloads the fixed Food.com v2 source, which is approximately 170 MB, and
regenerates `data/curated/recipes_2000.parquet`. No LLM API key is required, but allow at
least 1 GB of free disk space for the download, cache, and generated files.

On Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[data]"
.\.venv\Scripts\python.exe scripts\download_source.py
.\.venv\Scripts\python.exe scripts\inspect_source.py
.\.venv\Scripts\python.exe scripts\curate_corpus.py
```

On macOS or Linux, use `./.venv/bin/python` in place of
`.\.venv\Scripts\python.exe`.

The scripts use a fixed Kaggle dataset version and random seed. They also generate a source
schema, record manifest, deterministic QA sample, data card, and curation summary. Do not
run these commands unless you intentionally want to rebuild the provided corpus.
`curate_corpus.py` replaces the included `recipes_2000.parquet`; stop Streamlit before
running it and restart the application after the rebuild finishes.

## Repository structure

```text
app.py                         Streamlit user interface
src/fridge_rescue/             retrieval, model calls, validation, and workflow
src/fridge_rescue/data_pipeline.py
                               reusable parsing, cleaning, and validation functions
scripts/download_source.py     download and verify the fixed Kaggle source
scripts/inspect_source.py      inspect the source schema and data quality
scripts/curate_corpus.py       clean, deduplicate, sample, and write 2,000 recipes
data/curated/recipes_2000.parquet
                               local recipe corpus required by the application
config.yaml                    retrieval thresholds
rules.yaml                     ingredient aliases and deterministic rules
prompts.yaml                   model role and repair prompts
.streamlit/config.toml         interface theme
.env.example                   safe configuration template without secrets
```

## Data source

The curated corpus is derived from the Food.com Recipes and Reviews dataset on Kaggle,
version 2 (`irkaal/foodcom-recipes-and-reviews`). The source dataset is CC0 1.0 Public
Domain.
