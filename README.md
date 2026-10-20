# Fridge Rescue

Fridge Rescue is a Streamlit recipe application that turns available ingredients into
practical recipe suggestions. The repository is intentionally limited to three things:

1. the code required to run the application;
2. the reproducible Epicurious data-processing pipeline; and
3. the prepared 2,000-recipe corpus used directly by the application.

Development tests, formal experiment runners, tuning utilities, presentation files and
experiment outputs are not included in this public runtime repository.

- Live app: <https://fridge-rescue-ai.streamlit.app/>
- Repository: <https://github.com/Kevin520520/fridge-rescue>

## Application modes

- **Local retrieval** — searches the prepared corpus without calling an external model.
- **Single-prompt** — asks the configured model to generate recipes in one call.
- **Agentic workflow** — retrieves local candidates, lets the model select recipe IDs,
  validates the selection and displays the original corpus quantities and instructions. If
  both model drafts fail validation, it safely falls back to the validated local ranking.

The interface accepts common English and supported Chinese ingredient names. Matched
ingredients are highlighted in the returned recipe cards.

## Repository contents

```text
app.py                               Streamlit interface and mode routing
src/fridge_rescue/                   Application and data-pipeline modules
config.yaml                          Corpus and retrieval settings
rules.yaml                           Ingredient aliases and curation rules
prompts.yaml                         Single-prompt and Agentic prompt contracts

data/curated/recipes_2000.parquet    Prepared dataset used directly by the app
data/curated/curation_summary.json   Source, filtering and release statistics
data/curated/DATA_CARD.md            Dataset scope, licence and limitations

scripts/download_source.py           Download and verify the pinned source
scripts/inspect_source.py            Inspect source schema and quality
scripts/curate_corpus.py             Clean, deduplicate and sample 2,000 recipes
scripts/audit_curated_corpus.py      Audit every generated recipe record
scripts/check_setup.py               Check runtime and rebuild prerequisites
```

## Run locally on Windows

Install Python 3.11, 3.12 or 3.13, then open PowerShell:

```powershell
git clone https://github.com/Kevin520520/fridge-rescue.git
cd fridge-rescue
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . -c requirements-lock.txt
.\.venv\Scripts\python.exe scripts\check_setup.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

For Python 3.12 or 3.11, replace `-3.13` in the environment-creation command. Streamlit
normally opens <http://localhost:8501>. If it asks for an email address on first launch,
leave the field blank and press Enter.

The prepared Parquet file is already included, so normal users do **not** need to run the
data pipeline.

### Quick local test

Choose **Local retrieval** and enter:

```text
egg, tomato
```

or:

```text
鸡蛋，番茄
```

## Enable the AI modes

Local retrieval works without an API key. To enable Single-prompt and Agentic modes:

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace the placeholder with your own DeepSeek API key, save the file and restart
Streamlit. Never commit `.env` or publish a real API key.

## Prepared dataset

The current corpus is derived from
[Epicurious Recipes (13k)](https://github.com/josephrmartinez/recipe-dataset) at revision
`c5a1e8dab69d46841d914be1f20cca6fbe4b6260`.

```text
13,501 source rows
 7,578 eligible recipes after validation and deduplication
 2,000 released recipes selected with fixed seed 6012
```

Each released recipe preserves the source ingredient lines, quantities, instructions,
source identifiers and licence fields. Normalized fields are added for matching; they do
not replace the original recipe text.

## Rebuild the 2,000-recipe corpus

After installing the project dependencies, run:

```powershell
.\.venv\Scripts\python.exe scripts\download_source.py
.\.venv\Scripts\python.exe scripts\inspect_source.py
.\.venv\Scripts\python.exe scripts\curate_corpus.py
.\.venv\Scripts\python.exe scripts\audit_curated_corpus.py
.\.venv\Scripts\python.exe scripts\check_setup.py --check-rebuild
```

This process:

1. downloads the fixed 26.6 MB source CSV;
2. verifies its file size and SHA-256;
3. inspects the required source columns;
4. rejects incomplete or corrupted records;
5. deduplicates and normalizes matching fields;
6. performs deterministic category/complexity sampling; and
7. writes and audits the 2,000-recipe release.

The raw CSV and rebuild-only outputs are generated locally and intentionally excluded from
GitHub. The data pipeline does not require an LLM API key, but the first download requires
internet access.

Running `scripts/curate_corpus.py` writes the release files into `data/curated/`. Use a
fresh clone if you want to compare a rebuilt release with the packaged copy.

## Source and licence

The pinned source and the derived corpus are distributed under **CC BY-SA 3.0**. Source
revision, SHA-256, curation thresholds and release statistics are recorded in
`data/curated/curation_summary.json` and `data/curated/DATA_CARD.md`.
