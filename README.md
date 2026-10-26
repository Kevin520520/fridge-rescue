# Fridge Rescue

Fridge Rescue is a Streamlit recipe application that turns available ingredients into
practical recipe suggestions. The repository is intentionally limited to three things:

1. the code required to run the application;
2. the reproducible Epicurious data-processing pipeline; and
3. the prepared 2,000-recipe corpus used directly by the application.

Formal experiment runners, tuning utilities, presentation files and
experiment outputs are not included in this public runtime repository.

- Live app: <https://fridge-rescue-ai.streamlit.app/>
- Repository: <https://github.com/Kevin520520/fridge-rescue>

## Application modes

- **Local retrieval** — searches the prepared corpus without calling an external model.
  Optional photo recognition and natural-language parsing are separate AI tools and call
  the configured provider only when clicked.
- **Single-prompt** — asks the configured model to generate recipes in one call.
- **Agentic workflow** — retrieves local candidates, lets the model select recipe IDs,
  validates the selection and displays the original corpus quantities and instructions. If
  both model drafts fail validation, it safely falls back to the validated local ranking.

The interface accepts common English and supported Chinese ingredient names. Matched
ingredients are highlighted in the returned recipe cards.
Application-owned text uses English or bilingual English/Chinese wording throughout,
including graph labels, table headers, time-estimate evidence and AI clarification
questions. Clarification output requires separate English and Chinese fields.

### Cooking filters and natural-language requests

The search panel supports a time budget (30 minutes by default when enabled),
vegetarian/vegan preferences, ingredient exclusions, and available special equipment.
Equipment filtering covers the six corpus tags: oven, microwave, blender/food processor,
slow cooker, grill and waffle iron. Basic pots, pans and a stovetop are assumed available.
These filters are shared by local retrieval, Agentic recommendations and graph recipe
queries. The single-prompt comparison receives the same structured request.

Open **Describe your cooking plan · 描述你的烹饪计划** and try:

```text
有鸡蛋和番茄，30 分钟内，蛋奶素，没有烤箱，不要花生。
```

Click **解析需求**, review the extracted ingredients and conditions, then click
**确认并填入**. Ingredients/exclusions merge with existing entries; explicitly stated
conditions update the editable filters. Unmentioned conditions stay as entered.
Ambiguous or unsupported constraints produce clarification questions and cannot be
applied until the description is resolved. Changing the sentence clears its old parse.
Parsing makes one Structured Output call using the existing provider key; it does
not run automatically while typing.

The released source has no total-time values and very few explicit diet labels.
At runtime, `cooking_metadata.py` adds ingredient-rule diet labels and a separate,
visible **step-based time estimate**, preserving the original source fields. The
deterministic estimate uses the upper bound of a required duration range, selects the
shorter explicitly alternative path, and includes per-side times and mandatory waits.
It then adds 10 preparation minutes, 2 minutes per step and 5 minutes per untimed cooking
step. Optional storage, make-ahead extensions and alternative overnight waits are
excluded when the instruction wording identifies them.
Because recipe prose is not fully standardized, this remains an approximate filter rather
than a guaranteed cooking time. Cards show **步骤估算** and their Source tab lists the
calculation. Uncheck **包含步骤估算时长** to require a source-provided total time.
Diet rules check animal markers, plant dairy exceptions, recognized ingredients and
ambiguous prepared products; unresolved recipes retain an unknown diet status and
are excluded by a diet filter. These are deterministic supporting rules, not a trained
prediction model.

### Identify ingredients from multiple photos

Upload 1–5 JPG, PNG or WebP photos (up to 10 MB and 25 megapixels each, 25 MB total), then click
**Recognize ingredients / 识别食材**. The app shows the identified ingredient types
with Chinese and English names. Select the names you want and click **Use selected
ingredients / 使用所选食材** to add them to the ingredient field. Existing entries are
preserved, and configured English/Chinese aliases are deduplicated across all photos.
Fridge, vegetable and seasoning photos are sent together in one model call and produce
one combined list of up to 40 ingredient types. Invalid images must be removed or
replaced before the batch can run. You can also use
the photo feature just to read the ingredient names, without searching for recipes.

Recognition uses the active provider's API key and base URL. With the default DeepSeek
configuration it uses `deepseek-flash`, which supports
[image input](https://api-docs.deepseek.com/guides/vision/). With OpenAI it uses
`OPENAI_MODEL`; `VISION_MODEL` can override the image model for either provider.
The selected model must support image input and Structured Outputs.

Photos are decoded, oriented, resized to at most 1600 pixels per side and re-encoded
without the original metadata before the recognition request. The app keeps photos and
recognition results in the user's current session and does not write them to disk.
Only clicking **Recognize ingredients** sends photos to the provider; changing recipe
inputs does not trigger another recognition request. Removing or replacing the upload
clears the previous recognition result. Without an API key, manual ingredient entry
and local recipe retrieval remain available.

### Ingredient graph and query tools

Open **食材关系图与图查询** to use these local, read-only tools:

- **Recipe paths** traverses ingredient aliases to corpus ingredient/recipe relationships.
  It can require every queried ingredient and applies the current cooking filters. The
  graph and table show the matching source ingredient lines and source record IDs.
- **Co-occurrence** counts distinct recipes containing both foods. Quantity variants
  with known canonical names are combined; each recipe contributes once per ingredient.
  Every result includes an example recipe and source ID.
- **Reviewed substitutions** follows the directional substitution rules in `rules.yaml`,
  returning their allowed contexts and warnings. Replacement suggestions do not rewrite
  recipe facts or count as ingredients the user owns.

The graph is built from the 2,000 local recipes plus the existing alias and substitution
rules. It requires no graph server, extra key or external model call. Co-occurrence is
an ingredient → recipe ← ingredient traversal over corpus evidence.

The Agentic LangGraph workflow now executes:

```text
validate_and_normalize → query_ingredient_graph → retrieve_and_rank
    → generate_answer → validate_answer → [repair_once → validate_answer]
```

The graph tool limits the candidate IDs before deterministic filtering/ranking; the
model then selects only from those candidates. The workflow outcome includes the
graph candidate count, ingredient evidence paths and node trace. Returned recipe cards
also offer a graph view of why their ingredients match. The domain ingredient graph
and LangGraph's orchestration graph serve separate roles.

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

The repository keeps tracked text files in LF format. The setup checker also normalizes
LF/CRLF line endings when verifying `rules.yaml`, so Windows Git settings do not produce
a false data-integrity warning.

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

The same key also enables optional photo recognition and natural-language parsing.
Existing Streamlit deployments can reuse their configured provider secret. No separate
vision key is required.

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
