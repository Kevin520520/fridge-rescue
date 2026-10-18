# Fridge Rescue

Fridge Rescue is a bilingual Agentic AI recipe recommender. It searches a curated local
collection of 2,000 Food.com recipes and can optionally use DeepSeek or OpenAI to select
and validate recommendations.

> The processed 2,000-recipe corpus is already included. Normal users do not need to
> download or process the Kaggle source again. An API key is also optional because the
> local retrieval mode works offline.

## 在线部署给组员体验（Streamlit Community Cloud）

部署完成后，组员只需打开生成的 `*.streamlit.app` 链接，不需要安装Python或下载代码。

1. 登录 [Streamlit Community Cloud](https://share.streamlit.io/) 并点击 **Create app**。
2. 选择以下配置：
   - Repository：`Kevin520520/fridge-rescue`
   - Branch：`main`
   - Main file path：`app.py`
3. 打开 **Advanced settings**，Python选择 `3.13`。
4. 如需体验Agentic和Single-prompt模式，在 **Secrets** 中填写：

   ```toml
   LLM_PROVIDER = "deepseek"
   DEEPSEEK_API_KEY = "替换为自己的DeepSeek API Key"
   DEEPSEEK_MODEL = "deepseek-flash"
   DEEPSEEK_BASE_URL = "https://api.deepseek.com"
   ```

5. 点击 **Deploy**，等待构建完成后，将应用链接发给组员。

不要把真实API Key写进仓库、`.env.example`、README、聊天或截图。公开应用的所有AI
请求都会消耗部署者的API额度；只供小组使用时，建议使用私有分享，或在演示结束后移除
Secret。没有配置API Key时，云端应用仍可使用本地检索模式。

## 中文 Windows 快速开始（推荐）

### 1. 下载并解压项目

最简单的方法：

1. 点击GitHub页面右上方的 **Code**。
2. 选择 **Download ZIP**。
3. 下载完成后右键ZIP文件，选择**全部解压**。不要直接在压缩包中运行。
4. 打开VS Code，选择 **File → Open Folder**，打开解压后的 `fridge-rescue-main`
   文件夹。正确的文件夹中应该能看到 `app.py` 和 `pyproject.toml`。
5. 在VS Code中选择 **Terminal → New Terminal**。

如果已经安装Git，也可以在PowerShell中运行：

```powershell
git clone https://github.com/Kevin520520/fridge-rescue.git
cd fridge-rescue
```

### 2. 检查Python

在VS Code终端中运行：

```powershell
py --version
py -0p
```

项目支持Python 3.11、3.12和3.13。如果显示“无法识别 `py`”，请先从
[python.org/downloads](https://www.python.org/downloads/) 安装上述版本之一，安装时勾选
**Add python.exe to PATH**，
然后完全关闭并重新打开VS Code。

### 3. 安装并启动应用

下面四条命令需要逐条运行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run app.py
```

如果电脑安装的是Python 3.12或3.11，把第一条命令中的 `-3.13` 分别改成
`-3.12` 或 `-3.11`，其余命令不变。

第一次启动Streamlit时，终端可能询问 `Email:`。可以直接留空并按Enter。
随后打开终端显示的 **Local URL**，通常是 `http://localhost:8501`。如果显示其他
端口，以终端实际地址为准。运行期间不要关闭终端。

### 4. 用30秒确认部署成功

没有配置API Key时，页面只显示 **Local retrieval · 本地检索**，这是正常现象。

1. 在“可用食材”中输入 `egg, tomato, cooked rice`，或输入 `鸡蛋，番茄，米饭`。
2. 点击 **Find recipes**。
3. 页面能够显示菜谱标题、食材和步骤，即表示本地部署成功。
4. 回到终端按 `Ctrl+C` 可以关闭应用。

### 5. 可选：开启Agentic和Single-prompt模式

离线模式不需要API Key。只有需要体验AI模式时才执行：

```powershell
Copy-Item .env.example .env
notepad .env
```

把 `.env` 中的 `replace_with_your_own_key` 替换为自己的DeepSeek API Key，保存并
关闭记事本。回到终端按 `Ctrl+C` 停止应用，然后重新运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

重新打开页面后，应当可以选择 **Smart recommendation · 智能推荐**、
**Single-prompt AI · 单次 AI** 和 **Local retrieval · 本地检索**。不要把 `.env`
上传到GitHub，也不要在聊天或截图中分享API Key。

### 常见问题

- **提示无法识别 `py`：** Python尚未安装或没有加入PATH；安装后重新打开VS Code。
- **提示找不到 `app.py` 或 `pyproject.toml`：** 当前终端不在项目根目录；应进入同时
  包含这两个文件的文件夹。
- **只显示本地检索模式：** 这是没有有效API Key时的正常表现；离线功能仍可使用。
- **网页没有自动打开：** 手动打开终端显示的Local URL。
- **8501端口被占用：** Streamlit通常会显示另一个可用端口，请使用终端实际地址。
- **关闭终端后网页失效：** 应用依赖本地终端进程，需要重新运行最后一条启动命令。

## English quick start

Python 3.11, 3.12, or 3.13 is required. Clone the public repository and run the following
commands from the project root:

### Windows PowerShell

```powershell
git clone https://github.com/Kevin520520/fridge-rescue.git
cd fridge-rescue
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Use `-3.12` or `-3.11` in the first command when appropriate.

### macOS or Linux

```bash
git clone https://github.com/Kevin520520/fridge-rescue.git
cd fridge-rescue
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m streamlit run app.py
```

Without an API key, the application automatically offers the offline retrieval mode.
To enable live AI modes, copy `.env.example` to `.env`, add a personal DeepSeek or OpenAI
key, and restart Streamlit.

## Application modes

- **Local retrieval:** searches the included corpus without an external model or API cost.
- **Agentic AI workflow:** retrieves candidates, lets the model select recipe IDs, validates
  the output, and performs at most one repair.
- **Single-prompt comparison:** calls the same configured model without the local corpus or
  Agentic workflow.

Agentic and single-prompt modes use the same configured provider and model. This keeps the
course comparison focused on the value of retrieval and orchestration.

## Optional: rebuild the 2,000-recipe corpus

This section reproduces the data preparation process and is not required for normal use.
It downloads the fixed Food.com v2 source, which is approximately 170 MB, and regenerates
`data/curated/recipes_2000.parquet`. No LLM API key is required, but allow at least 1 GB of
free disk space for the download, cache, and generated files.

Stop Streamlit first, then run on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[data]"
.\.venv\Scripts\python.exe scripts\download_source.py
.\.venv\Scripts\python.exe scripts\inspect_source.py
.\.venv\Scripts\python.exe scripts\curate_corpus.py
```

On macOS or Linux, use `./.venv/bin/python` in place of
`.\.venv\Scripts\python.exe`.

The scripts use a fixed Kaggle dataset version and random seed. They download and verify
the source, inspect its schema, clean and deduplicate valid records, apply deterministic
stratified sampling, and generate the final 2,000-recipe corpus. They also write a source
schema, manifest, QA sample, data card, and curation summary.

`curate_corpus.py` replaces the included `recipes_2000.parquet`. Restart the application
after the rebuild finishes.

## Repository structure

```text
app.py                         Streamlit user interface
src/fridge_rescue/             retrieval, model calls, validation, and workflow
src/fridge_rescue/data_pipeline.py
                               reusable parsing, cleaning, and validation functions
scripts/download_source.py     download and verify the fixed Kaggle source
scripts/inspect_source.py      inspect source structure and data quality
scripts/curate_corpus.py       clean, deduplicate, sample, and write 2,000 recipes
data/curated/recipes_2000.parquet
                               processed local corpus used directly by the application
config.yaml                    retrieval thresholds and fixed corpus settings
rules.yaml                     ingredient aliases and deterministic curation rules
prompts.yaml                   model roles and repair prompts
.streamlit/config.toml         interface theme
.env.example                   safe configuration template without secrets
```

## Data source

The curated corpus is derived from the Food.com Recipes and Reviews dataset on Kaggle,
version 2 (`irkaal/foodcom-recipes-and-reviews`). The source dataset is CC0 1.0 Public
Domain. The raw source is intentionally not stored in this repository.
