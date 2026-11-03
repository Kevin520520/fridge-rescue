# Fridge Rescue — UI alternatives for discussion

**Status: proposal only / 仅供备选，未实施**

**Contributor:** [@yahoovo](https://github.com/yahoovo)

**Date:** 2026-11-03

**Review baseline:** `staging` at `87c851bfad5cfc0fb1e8211023bcf7365d19f2ea`

This document proposes three feasible UI directions for the existing Streamlit app. It changes no application code, configuration, data, dependencies, or deployment settings. The layouts below are sketches, not implemented screens or usability-test results. No option has been approved for implementation.

本文提出三个基于现有 Streamlit 的 UI 备选方向，仅供团队讨论。所有布局均为草图；成本为初步估计，效果需要实际验证。

## 1. What the current implementation already provides

The code review found these opportunities. They are design hypotheses, not measured user problems.

| Existing behavior | Opportunity | Source at the reviewed revision |
| --- | --- | --- |
| The large hero uses gradients, shadow and decorative food emoji. Every button receives the same orange gradient. | Establish a quieter editorial identity and distinguish the primary search action from supporting actions. | [`app.py`](../../app.py), `_inject_app_styles()` |
| Recommendation mode, photo input and language parsing appear before the manual ingredient field. The photo panel starts expanded. | Give the main ingredient-to-recipe task a clearer starting point. | [`app.py`](../../app.py); [`image_recognition_ui.py`](../../src/fridge_rescue/image_recognition_ui.py), `render_image_ingredient_input()` |
| Time, diet and equipment controls sit in the same input flow, with an ingredient graph tool before search. | Group optional detail while keeping active restrictions visible. | [`cooking_constraints_ui.py`](../../src/fridge_rescue/cooking_constraints_ui.py), `render_cooking_filters()`; [`app.py`](../../app.py) |
| Result cards already show ingredient use, missing items, estimated time, and Ingredients / Steps / Source tabs. | Improve scanning using existing evidence, without adding unsupported facts. | [`app.py`](../../app.py), `_display_recipe()` |
| AI failure, no candidates and validation fallback have distinct outcomes. | Make the next available action clearer while preserving those distinctions. | [`app.py`](../../app.py), `_display_workflow_outcome()` |

## 2. Option A — Editorial kitchen / 杂志式厨房

**Best fit:** a distinctive, calm visual identity with modest implementation effort.

Use a compact masthead, generous spacing, a serif display heading, and system sans-serif controls. A warm paper background and one dark tomato-red action color can replace the multiple gradients and repeated shadows. Recipe names and ingredient evidence carry the visual interest; photographs are optional and must not be fabricated as source-recipe images.

```text
FRIDGE RESCUE                         Cook with what you have
───────────────────────────────────────────────────────────
What can you make tonight?
用现有食材，找到今晚适合做的菜。

Available ingredients · 可用食材
[ eggs, tomatoes, rice                                     ]
Avoid · 避开 [                     ]  Time / Diet [controls]
[Optional: photos or a cooking description]
Current mode [selector]          [ Find recipes · 查找菜谱 ]
───────────────────────────────────────────────────────────
Tonight's picks / 今晚推荐
01  [Source recipe title]
    Uses X/Y input ingredients · N additional ingredients
    About T min — step estimate
    [Ingredients] [Steps] [Source]
```

**Implementation path:** revise `_inject_app_styles()` and the hero markup in `app.py`; use `type="primary"` only for the main search action and an appropriate secondary style for supporting actions. Scope any CSS narrowly, retain visible focus indicators, and check selectors against the installed Streamlit version. Keep existing retrieval and result functions.

**Trade-off:** fastest route to a more considered identity, but typography alone does not simplify the number of controls. System serif fallbacks avoid a new font dependency; Chinese text needs its own tested fallback.

**Initial effort:** roughly 0.5–1.5 developer days including responsive review, assuming no backend changes. This is an estimate, not a commitment.

## 3. Option B — Guided cooking / 分步烹饪助手

**Best fit:** first-time users who need a clear path from ingredients to results.

Use three visible sections on one page rather than a new multi-page router. Manual entry is the first path; photo recognition and natural-language parsing remain optional. Borrow the clarity of familiar mobile settings groups while retaining native Streamlit widgets.

```text
1  Your ingredients / 你有什么
   [Manual input]
   [Optional photo recognition] [Optional description parsing]
   Recognition or parsing → review → explicitly apply

2  Your cooking conditions / 烹饪条件
   Time [ ]   Diet [ ]   Avoid [ ]
   More conditions ▸ equipment and time-estimate policy
   Active conditions: [read-only summary of current settings]

3  Find a meal / 找一道菜
   Mode [Local retrieval / Agentic / Single-prompt]
   [Find recipes]
   [Results, or an explanation with the next available action]
```

**Implementation path:** reorganize the render order in `app.py`; keep `initialize_cooking_inputs()` before widgets and retain the existing session-state keys. Group optional controls in native expanders; render controls on every rerun so hiding a section does not discard its values. Build the active-condition summary from the same state passed to `request_from_inputs()`.

Keep explicit confirmation after recognition/parsing. Switching sections or typing must not trigger a model call. Keep the mode selector and provider notice visible before submission; do not silently change the selected mode. Keep ingredient exclusions visible even if other conditions are collapsed.

**Trade-off:** clearer hierarchy, but grouping can hide constraints unless the active-condition summary is always visible. A true multi-page wizard would require additional state and navigation work and is outside this proposal.

**Initial effort:** roughly 1–2 developer days, with particular attention to state persistence, photo review, and parsed-condition updates.

## 4. Option C — Decision-first recipe cards / 决策优先菜谱卡

**Best fit:** users who can enter ingredients but need help comparing the returned options.

Keep the input flow largely intact. Give each result the same compact decision summary, followed by expandable detail. Use a vertical stack on all screen sizes as the baseline; a desktop comparison grid is optional after readability testing.

```text
[Source recipe title]
Uses X/Y of your input ingredients
Also needed: [actual missing ingredients]
About T min · step estimate [calculation in Source]
Why it matches: [existing evidence-based explanation]

[Ingredients · 食材] [Steps · 步骤] [Source · 来源]
```

**Implementation path:** adjust `_display_recipe()` and the presentation of `_display_workflow_outcome()`. Reuse `_coverage_summary()` and `_requested_ingredient_summary()`; do not invent a match score. Keep the original source quantities and steps, time-estimate qualification, and evidence access. Baseline-generated recipes in `_display_baseline_outcome()` must retain their separate provenance labels.

For no results, show the current conditions and suggest editing ingredients or time. Never remove exclusions, diet or equipment constraints automatically. For an AI service error, offer an explicit return to Local retrieval with the inputs retained. For validation fallback, preserve the existing disclosure that the displayed results came from local ranking.

**Trade-off:** improves comparison without new retrieval logic, but does less to simplify first-time input. Source-only time can exclude every record when source times are absent; the explanation must make this understandable.

**Initial effort:** roughly 0.5–1.5 developer days including checks for missing metadata and different workflow outcomes.

## 5. Suggested direction for team review

Start by discussing **A's visual treatment + B's single-page grouping**. Treat C as a separate, small result-card refinement. This combines the requested editorial aesthetic with straightforward mobile interaction, while staying within the current Streamlit stack.

| Decision | Suggested starting point | Reason |
| --- | --- | --- |
| Visual tone | Warm paper, dark text, one tomato-red accent | Relates to cooking while reducing visual competition |
| Typography | System serif for one short heading; system sans-serif for controls/body | Editorial character without remote font loading |
| Spacing | 8 / 16 / 24 / 32 px rhythm; approximately 960 px content maximum | Consistent grouping within the existing page width |
| Shape | Moderate 8–12 px corner radii; thin borders; minimal shadow | Gives recipe content more emphasis than containers |
| Language | Keep English/Chinese labels together; English title with Chinese support line | Preserves the existing bilingual behavior |
| Navigation | One page, native widgets and optional expanders | Limits state-management and maintenance cost |

Illustrative palette: paper `#FAF7F0`, text `#242923`, secondary text `#596157`, primary action `#9A3412` with white text, matched-ingredient text `#28522B` on `#DFF1D8`. These are proposal tokens; verify contrast for actual combinations before implementation. Color must not be the only sign of ingredient matching, errors or selection.

## 6. Acceptance checks for a future implementation

These checks are proposed work, not completed validation of a redesigned app.

- At 375, 768 and 1440 px widths, ingredient input and the primary action remain readable without horizontal scrolling; bilingual labels wrap without clipping.
- All inputs have visible labels. Keyboard order follows the visual order; focus stays visible; interactive targets are at least 44 × 44 px where practical. Normal text reaches 4.5:1 contrast.
- Existing English/Chinese alias handling, exclusions and constraint logic remain unchanged. The same request produces the same local recipe IDs and ranking before and after the UI change.
- Manual entry and Local retrieval work without an AI key. Photo recognition and language parsing run only after an explicit click and require review before applying results.
- Opening optional sections, editing inputs and switching modes retain intended input values. A parsed request updates only the fields it explicitly provides, as in the current implementation.
- Recipe cards preserve source quantities, original instructions and provenance. Estimated times stay labelled as estimates; unknown values do not become invented ratings or durations.
- Empty input, no candidates, service errors and validation fallback are distinguishable. Suggestions never silently loosen user constraints.
- If results are retained across reruns in a future implementation, label them with the submitted request and mark them stale after inputs change. This would require explicit state handling, not merely CSS.

## 7. Review and scope

The proposed contribution is this document only. Team feedback can select an option, combine specific elements, or keep the current UI. Any implementation should be a separate change with its own review and validation.

Discussion prompts: Which option best fits the demonstration? Is manual input the preferred first path? Which constraints must stay visible? Is the proposed editorial tone appropriate for the team?

**中文摘要：** A 调整杂志式视觉；B 整理“食材—条件—推荐”的操作顺序；C 优化菜谱比较与异常提示。建议优先讨论 A+B，C 可独立采纳。本次仅提交方案文档，不修改、不上线任何正式 UI。
