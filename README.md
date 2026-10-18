# Fridge Rescue

Fridge Rescue is a bilingual recipe recommender for ingredients already available at home.
It combines a curated local collection of 2,000 recipes, deterministic retrieval, a
LangGraph workflow, and an optional DeepSeek or OpenAI model.

This repository is the minimal runnable version. Development tests, experiment outputs,
raw source data, audit archives, and corpus-building utilities are intentionally excluded.

## What is included

- **Offline retrieval preview:** works without an API key.
- **Agentic AI workflow:** retrieves recipe candidates, lets the model select recipe IDs,
  validates the result, and performs at most one repair.
- **Single-prompt comparison:** calls the same configured model without the local corpus or
  workflow.
- **Curated corpus:** 2,000 Food.com recipes stored locally in Parquet format.
- **Chinese ingredient aliases:** a reviewed set of common Chinese ingredient names is
  normalized to the English corpus vocabulary.

## Requirements

- Python 3.11, 3.12, or 3.13
- Internet access only for installing packages and using a live AI mode
- A personal DeepSeek or OpenAI API key for live AI modes

## Run on Windows

Open PowerShell in the cloned project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will show a local address such as `http://localhost:8501`. Open it in a browser.
Without an API key, the application automatically offers the offline retrieval preview.

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
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m streamlit run app.py
```

## Repository structure

```text
app.py                         Streamlit user interface
src/fridge_rescue/             retrieval, model calls, validation, and workflow
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
