# Fridge Rescue

Fridge Rescue is a bilingual recipe recommender built for the HD6012 Agentic AI
group assignment. Users enter ingredients already at home and optional ingredients
to avoid. The application can search a curated local corpus, ask an AI model to
select grounded recipes, validate the result, and display the original quantities
and instructions from the dataset.

- GitHub: https://github.com/Kevin520520/fridge-rescue
- Live app: https://fridge-rescue-ai.streamlit.app/

## 三种使用模式

- **Agentic AI workflow / 智能推荐**：本地检索 2,000 条菜谱，AI 只从候选
  `recipe_id` 中选择，程序检查 ID、食材覆盖和限制条件；失败时最多修复一次。
- **Single-prompt AI / 单次 AI**：同一个 DeepSeek 模型只接收一次用户提示，不使用
  本地语料、检索、验证或修复；这是课程实验的 baseline。
- **Local retrieval / 本地检索**：只运行本地确定性检索，不调用外部 AI，也不消耗
  API 额度。

Agentic 和 Single-prompt 使用同一个 provider 和 model，因此比较的是“本地语料 +
检索 + 编排 + 验证”带来的差异，而不是两个不同大模型之间的差异。

## 当前数据版本：Corpus v3.1

当前数据来自固定版本的
[Epicurious Recipes (13k)](https://github.com/josephrmartinez/recipe-dataset)：

- 固定源文件：13,501 条，26,595,455 bytes；
- 源文件 SHA-256：
  `d0e3977da7c1dc287ed75d6770f4064a7caeaa20ef6f64708b730e02f4d5f17a`；
- 通过确定性清洗的完整记录：7,578 条；
- 固定随机种子分层抽样后的发布语料：2,000 条；
- 最终语料包含 20,997 条食材行，删除或隐藏的食材行：0；
- 每条菜谱至少 3 条食材、2 个步骤，计量覆盖率最低 60%，平均 93.47%；
- 数量与单位只保留源数据内容，不由代码或 AI 猜测；
- 仅修复 16 条菜谱中的 18 处明确连写空格，例如 `11/2 cups` →
  `1 1/2 cups`，数值本身不变；
- 全量审计结果：0 个 hard failure，0 个 review warning；
- 最终 Parquet SHA-256：
  `1fbeeaa09544940884e64dc94a9b3666658a5956790c7f36487f26892371e690`。

如果一条源菜谱含有确认的损坏数量、损坏单位、非食材内容、未展开的子菜谱引用
或结构不完整内容，清洗程序会淘汰整条菜谱，而不是隐藏错误行后继续保留菜谱。
详细证据位于 `data/curated/CURATION_AUDIT_TRAIL.md`、
`data/curated/curation_summary.json` 和 `data/curated/audits/v31_20261018/`。

源数据是社区发布内容。上述检查可以保证结构、来源和计量规则一致，但不能证明
每一道菜都已经在厨房中实际测试过；项目不会作出这种过度声明。

## 核心架构

```text
用户输入
  → 中英文别名归一化与输入检查
  → 本地 2,000 条语料的确定性检索与排序
  → 从 Top-10 中确定前三个最强匹配候选
  → DeepSeek 只从这三个候选 ID 中选择（Agentic 模式）
  → 程序检查 ID、限制条件和输入食材利用率
  → 必要时最多修复一次
  → 根据 ID 从本地语料加载原始标题、数量、单位和步骤
```

默认检索先使用 `coverage >= 0.20`、额外食材不超过 7 的主门槛；只有主候选中
没有完整命中用户全部食材的菜谱时，才允许完整命中的救援候选使用
`coverage >= 0.15`、额外食材不超过 8。这个两级策略提升了完整食材命中率，
同时避免把大量低相关部分命中结果全部放宽进入候选池。

检索排序依次考虑：用户输入食材利用率越高越好、仍需购买的非基础食材越少越好、
菜谱侧覆盖率越高越好，最后才参考评分等质量信息。Agentic 选择器只接收 Top-10
中排序最强的前三条；程序还会验证模型至少选择一条同时满足前两个优先级的菜谱，
并按同一规则排列最终结果。

只有整行确实属于水、盐、黑胡椒等基础储备品时才不计入购物负担。例如
`low-salt chicken broth`、`rose water` 和 `water chestnuts` 仍会作为额外食材，
不会因为包含 `salt` 或 `water` 而被隐藏。菜谱食材页只绿色高亮实际命中的食材
词或词组，不再高亮整行，也不再显示额外的“已有”标签。

主界面只保留“可用食材”和“需要避开的食材”。源数据没有可靠时间字段，饮食标签
覆盖也过于保守，因此时间、设备和 vegan/vegetarian 没有作为当前 UI 的可靠筛选
卖点。应用也不提供医疗级过敏原或营养保证。

## 正式 Agentic vs Single-prompt 实验

冻结的 20 条正式案例分别运行两种模式，每种模式重复 3 次，共 120 次尝试：

| 指标 | Single-prompt | Agentic |
|---|---:|---:|
| 尝试次数 | 60 | 60 |
| API/输出错误 | 0 | 0 |
| 成功率（共同口径） | 100% | 95.0% |
| 已返回结果的硬约束通过率 | 100% | 100% |
| 适用案例的全部输入食材命中率 | 98.2% | 83.3% |
| 平均延迟 | 3963.6 ms | 1296.7 ms |
| Agentic 返回 ID 有效率 | 不适用 | 100% |

结果不支持“Agentic 在所有指标上都优于 Single-prompt”的说法。Single-prompt
更容易自由生成并使用全部输入食材；Agentic 的优势是结果可追溯到本地语料、返回
内容受验证、硬约束稳定，而且本轮平均延迟更低。Agentic 的主要不足来自语料覆盖：
真实世界有菜可做，不代表固定 2,000 条语料一定包含对应菜谱。

正式证据位于 `evaluation/results/formal_v33_20261018/`。这是在同一组 20 条已审核
案例上的版本回归比较，不应写成全新的盲测。两种模式均没有 API 或输出格式错误；
Agentic 也没有无效 ID 或验证错误。F18 是明确的本地语料缺口，F19 是输入与避开
食材相互矛盾，两种情况均按预期处理。正式集冻结后不再据此继续调参，避免在最终
测试集上过拟合。

## Windows 本地运行（组员推荐步骤）

先安装 Python 3.11、3.12 或 3.13，然后在 PowerShell 中执行：

```powershell
git clone https://github.com/Kevin520520/fridge-rescue.git
cd fridge-rescue
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\check_setup.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

如果电脑使用 Python 3.12 或 3.11，只需把第一条 Python 命令中的 `-3.13`
改成对应版本。第一次启动 Streamlit 时可以把 `Email:` 留空并按 Enter。
浏览器通常会打开 `http://localhost:8501`。

没有 API key 时仍可使用 **Local retrieval**。快速测试可输入：

```text
egg, tomato
```

或：

```text
鸡蛋，番茄
```

要启用两种 AI 模式：

```powershell
Copy-Item .env.example .env
notepad .env
```

把 `.env` 中的占位符换成自己的 DeepSeek API key，保存后重启 Streamlit。
真实 key 不得提交到 GitHub、README、截图或聊天中。

## 重新生成 2,000 条数据

普通使用者不需要重跑数据流程，因为处理好的 Parquet 已包含在仓库中。需要复现
数据准备时执行：

```powershell
.\.venv\Scripts\python.exe scripts\download_source.py
.\.venv\Scripts\python.exe scripts\inspect_source.py
.\.venv\Scripts\python.exe scripts\curate_corpus.py
.\.venv\Scripts\python.exe scripts\audit_curated_corpus.py
.\.venv\Scripts\python.exe scripts\check_setup.py --check-rebuild
```

这些脚本会下载固定 commit 的 26.6 MB CSV、验证文件哈希、检查源结构、淘汰整条
损坏记录、去重、分层抽样、生成 2,000 条语料并再次全量审计。数据流程不需要
LLM API key。

## 主要文件

```text
app.py                              Streamlit 界面与三种运行模式
src/fridge_rescue/workflow.py       LangGraph Agentic 工作流
src/fridge_rescue/retrieval.py      确定性检索、排序和两级门槛
src/fridge_rescue/openai_selector.py DeepSeek/OpenAI 结构化选择
src/fridge_rescue/evaluation.py     双模式评估指标
src/fridge_rescue/epicurious_pipeline.py
                                    当前 Epicurious 数据清洗规则
scripts/download_source.py          下载并校验固定源文件
scripts/inspect_source.py           检查源数据结构和质量
scripts/curate_corpus.py            清洗、去重、分层抽样和生成发布文件
scripts/audit_curated_corpus.py      对最终 2,000 条进行全量审计
scripts/check_setup.py               验证依赖、文件数量、哈希和可复现性
data/curated/recipes_2000.parquet   App 直接使用的新数据集
rules.yaml                          中英文别名、清洗和安全规则
config.yaml                         检索门槛和实验数量
prompts.yaml                        Agent 角色与修复提示词
```

## Source and license

The current corpus is derived from
[Epicurious Recipes (13k)](https://github.com/josephrmartinez/recipe-dataset) at
revision `c5a1e8dab69d46841d914be1f20cca6fbe4b6260`. The pinned source and the
derived corpus are distributed under CC BY-SA 3.0. The raw CSV is not committed;
the downloader retrieves and verifies the exact pinned file.
