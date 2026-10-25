"""Explore corpus ingredient paths, co-occurrence and reviewed substitutions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st
from pydantic import ValidationError

from fridge_rescue.cooking_constraints import request_from_inputs
from fridge_rescue.ingredient_graph import RecipeGraphEvidence, RecipeIngredientGraph
from fridge_rescue.input_normalization import parse_ingredient_text
from fridge_rescue.repository import ParquetRecipeRepository
from fridge_rescue.retrieval import DeterministicRecipeRetriever

QUERY_LABELS = {
    "recipes": "找菜谱和关系路径 · Recipe paths",
    "companions": "查共同出现的食材 · Co-occurrence",
    "substitutions": "查已审核替代规则 · Reviewed substitutions",
}


def _dot(edges: Sequence[tuple[str, str, str]], labels: Mapping[str, str]) -> str:
    """Quote every node ID and label so ingredient text cannot inject DOT syntax."""

    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    lines = ['digraph ingredients { rankdir=LR; graph [bgcolor="transparent"];',
             'node [shape=box, style="rounded,filled", fillcolor="#edf5e7", fontname="sans-serif"];']
    lines.extend(f"{quote(key)} [label={quote(label)}];" for key, label in labels.items())
    lines.extend(
        f"{quote(source)} -> {quote(target)} [label={quote(relation)}];"
        for source, target, relation in dict.fromkeys(edges)
    )
    return "\n".join([*lines, "}"])


def recipe_paths_dot(evidence: Sequence[RecipeGraphEvidence], titles: Mapping[str, str]) -> str:
    edges = []
    labels = {}
    for recipe in evidence:
        recipe_key = f"recipe:{recipe.recipe_id}"
        labels[recipe_key] = titles.get(recipe.recipe_id, recipe.recipe_id)
        for path in recipe.paths:
            query_key = f"query:{path.query_ingredient}"
            ingredient_key = f"ingredient:{path.corpus_ingredient}"
            labels[query_key] = path.query_ingredient
            labels[ingredient_key] = path.corpus_ingredient
            edges.extend([
                (query_key, ingredient_key, "Matches · 匹配食材"),
                (ingredient_key, recipe_key, "Used in · 用于菜谱"),
            ])
    return _dot(edges, labels)


def render_recipe_graph_evidence(
    graph: RecipeIngredientGraph, recipe_ids: Sequence[str], ingredients: Sequence[str],
    repository: ParquetRecipeRepository,
) -> None:
    """Show a compact evidence view beneath the returned recipe cards."""

    if not recipe_ids:
        return
    evidence = graph.explain(recipe_ids, ingredients, limit_per_recipe=4)
    titles = {recipe.recipe_id: recipe.title for recipe in repository.get_many(recipe_ids)}
    with st.expander("🕸️ Why these ingredients match · 查看食材关系路径"):
        st.graphviz_chart(recipe_paths_dot(evidence, titles), width="stretch")
        st.dataframe([
            {"Recipe · 菜谱": titles.get(item.recipe_id, item.recipe_id), "Recipe ID": item.recipe_id,
             "Input ingredient · 输入食材": path.query_ingredient, "Recipe ingredient · 菜谱食材": path.corpus_ingredient,
             "Original ingredient line · 原始食材行": path.source_line, "Source record · 源记录": item.source_record_id}
            for item in evidence for path in item.paths
        ], hide_index=True, width="stretch")


def render_ingredient_graph_query(
    graph: RecipeIngredientGraph, retriever: DeterministicRecipeRetriever,
    repository: ParquetRecipeRepository, values: Mapping[str, Any],
) -> None:
    with st.expander("🕸️ Ingredient graph tools · 食材关系图与图查询"):
        stats = graph.stats
        st.caption(
            f"Local graph: {stats['recipes']:,} recipes, {stats['ingredients']:,} ingredient nodes, "
            f"{stats['ingredient_edges']:,} ingredient relationships and {stats['substitutions']} reviewed substitution rules. / "
            f"本地图谱：{stats['recipes']:,} 道菜谱、{stats['ingredients']:,} 个食材节点、"
            f"{stats['ingredient_edges']:,} 条食材关系、{stats['substitutions']} 条已审核替代规则。"
        )
        query_type = st.selectbox(
            "Query tool · 查询工具", list(QUERY_LABELS), format_func=QUERY_LABELS.__getitem__,
            key="ingredient_graph_tool",
        )
        text = st.text_input(
            "Ingredients to explore · 想查询的食材", placeholder="e.g. egg, tomato or butter · 例如：鸡蛋、番茄或黄油",
            key="ingredient_graph_query", max_chars=300,
        )
        require_all = False
        if query_type == "recipes":
            require_all = st.checkbox("Must contain every queried ingredient · 必须同时包含全部查询食材", key="graph_require_all")
            st.caption("Recipe queries apply the time, diet, exclusions and equipment filters above, returning up to 5 recipes with source ingredient paths. / 菜谱查询使用上方时间、饮食、禁忌和厨具筛选项，返回最多 5 道菜谱及其原始食材路径。")
        elif query_type == "companions":
            st.caption("Enter one ingredient to count foods that share a recipe with it across the local collection, with counts and example sources. / 输入一种食材，统计整个本地语料中与它出现在同一道菜里的食材；显示计数和示例来源。")
        else:
            st.caption("Enter one ingredient to find reviewed replacements, allowed contexts and notes. Recipe searches use the original ingredients. / 输入一种食材，查询替代方向、适用情境和注意事项。替代规则用于参考，搜索按原菜谱食材匹配。")

        context = (
            "bilingual-v1", query_type, text, require_all, values.get("excluded_ingredients", ""),
            values.get("time_limit_enabled"), values.get("max_time_minutes"),
            values.get("allow_estimated_time"),
            values.get("diet_preference"), values.get("equipment_filter_enabled"),
            tuple(values.get("available_equipment", [])),
        )
        if st.session_state.get("ingredient_graph_context") != context:
            st.session_state.pop("ingredient_graph_result", None)
            st.session_state["ingredient_graph_context"] = context
        if st.button("Run graph query · 查询关系图", key="run_ingredient_graph_query", disabled=not text.strip()):
            st.session_state.pop("ingredient_graph_result", None)
            try:
                ingredients = parse_ingredient_text(text, graph.aliases)
                if not ingredients or len(ingredients) > 5:
                    raise ValueError("Enter 1–5 ingredients to query. / 请填写 1–5 种查询食材。")
                if query_type != "recipes" and len(ingredients) != 1:
                    raise ValueError("Enter one ingredient for co-occurrence or substitution queries. / 共现和替代查询每次填写一种食材。")
                rows = []
                dot = None
                if query_type == "recipes":
                    request = request_from_inputs(values, graph.aliases, ingredients_text=text)
                    found = graph.find_recipe_ids(ingredients, require_all=require_all)
                    recipe_ids = retriever.retrieve(request, limit=5, candidate_ids=found)
                    titles = {recipe.recipe_id: recipe.title for recipe in repository.get_many(recipe_ids)}
                    evidence = graph.explain(recipe_ids, ingredients, limit_per_recipe=5)
                    rows = [
                        {"Recipe ID": item.recipe_id, "Recipe · 菜谱": titles[item.recipe_id],
                         "Input ingredient · 输入食材": path.query_ingredient, "Recipe ingredient · 菜谱食材": path.corpus_ingredient,
                         "Original ingredient line · 原始食材行": path.source_line, "Source record · 源记录": item.source_record_id}
                        for item in evidence for path in item.paths
                    ]
                    dot = recipe_paths_dot(evidence, titles)
                elif query_type == "companions":
                    matches = graph.find_companions(ingredients[0])
                    rows = [
                        {"Co-occurring ingredient · 共同出现的食材": item.ingredient, "Shared recipe count · 共同菜谱数": item.recipe_count,
                         "Example Recipe ID · 示例菜谱编号": item.evidence_recipe_id, "Source record · 源记录": item.source_record_id}
                        for item in matches
                    ]
                    edges, labels = [], {"query": ingredients[0]}
                    for index, item in enumerate(matches):
                        key = f"companion:{index}"
                        labels[key] = item.ingredient
                        edges.append(("query", key, f"Co-occurs in {item.recipe_count} recipes · 共同出现在 {item.recipe_count} 道菜"))
                    dot = _dot(edges, labels)
                else:
                    matches = graph.find_substitutes(ingredients[0])
                    rows = [
                        {"Original ingredient · 原食材": item.original, "Possible substitute · 可考虑替代": item.substitute,
                         "Allowed contexts · 适用情境": ", ".join(item.allowed_contexts),
                         "Notes · 注意事项": item.warning, "Source · 来源": item.source}
                        for item in matches
                    ]
                    dot = _dot(
                        [(item.substitute, item.original, "Conditional replacement · 有条件替代") for item in matches],
                        {value: value for item in matches for value in (item.substitute, item.original)},
                    )
                st.session_state["ingredient_graph_result"] = {"rows": rows, "dot": dot}
            except (ValueError, ValidationError) as error:
                st.error(str(error))
        result = st.session_state.get("ingredient_graph_result")
        if result is not None:
            if not result["rows"]:
                st.info("No matching relationships were found. Adjust the ingredients or filters above. / 没有找到符合条件的关系。可以调整查询食材或上方筛选条件。")
            else:
                st.graphviz_chart(result["dot"], width="stretch")
                st.dataframe(result["rows"], hide_index=True, width="stretch")
