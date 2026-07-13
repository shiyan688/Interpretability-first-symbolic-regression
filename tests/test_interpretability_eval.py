from __future__ import annotations

import json

import pandas as pd
import pytest

from factor_pysr_llm.interpretability_eval import (
    RatingValidationError,
    build_judge_prompt,
    export_blind_ratings,
    load_rating_manifest,
    load_rubric,
    parse_rating_response,
    run_llm_judge,
    score_interpretability_prompt,
)

RUBRIC_PATH = "configs/interpretability_rubric.json"


def _candidates():
    return [
        {
            "formula": "v1 * v2",
            "variables": [{"name": "v1", "definition": "a", "unit": "x", "allowed_range": "[0,1]"}],
            "method": "if_sr",
            "seed": 1,
            "r2_test": 0.9,
            "dataset": "D1",
            "target": "t",
            "dataset_label": "dataset_A",
        },
        {
            "formula": "v1 + v2",
            "variables": [{"name": "v2", "definition": "b", "unit": "y", "allowed_range": "[0,2]"}],
            "method": "raw_pysr",
            "seed": 1,
            "r2_test": 0.88,
            "dataset": "D1",
            "target": "t",
            "dataset_label": "dataset_A",
        },
    ]


def test_blind_export_hides_private_fields(tmp_path):
    info = export_blind_ratings(
        _candidates(),
        out_manifest=tmp_path / "manifest.jsonl",
        out_private_map=tmp_path / "private.json",
        seed=7,
    )
    rows = load_rating_manifest(tmp_path / "manifest.jsonl")
    assert info["n_items"] == 2
    for row in rows:
        assert "method" not in row
        assert "r2_test" not in row and "r2" not in row
        assert "seed" not in row
        assert row["formula"]
    private = json.loads((tmp_path / "private.json").read_text())["mapping"]
    # private map retains method + r2
    methods = {v["method"] for v in private.values()}
    assert methods == {"if_sr", "raw_pysr"}
    assert (tmp_path / "manifest.human_template.csv").exists()


def test_blind_export_is_deterministic(tmp_path):
    a = export_blind_ratings(_candidates(), tmp_path / "m1.jsonl", tmp_path / "p1.json", seed=7)
    b = export_blind_ratings(_candidates(), tmp_path / "m2.jsonl", tmp_path / "p2.json", seed=7)
    ra = load_rating_manifest(tmp_path / "m1.jsonl")
    rb = load_rating_manifest(tmp_path / "m2.jsonl")
    assert [r["item_id"] for r in ra] == [r["item_id"] for r in rb]


def test_judge_prompt_embeds_formula_and_rubric(tmp_path):
    rubric = load_rubric(RUBRIC_PATH)
    item = {
        "item_id": "item_x",
        "dataset_label": "dataset_A",
        "formula": "v1 * v2 / v3",
        "variables": [{"name": "v1", "definition": "d1", "unit": "u", "allowed_range": "[0,1]"}],
        "task_context": "ctx",
    }
    prompt = build_judge_prompt(item, rubric)
    assert "v1 * v2 / v3" in prompt          # formula embedded
    assert "domain_meaning" in prompt         # rubric embedded
    assert "d1" in prompt                     # variable dict embedded
    assert "item_x" in prompt
    assert ".json" not in prompt.split("schema")[0] or "output_schema" not in prompt  # no local path reliance


def test_parse_valid_response():
    raw = json.dumps(
        {
            "item_id": "item_x",
            "ratings": {
                "domain_meaning": 4,
                "structural_plausibility": 3,
                "readability_generalizability": 5,
                "hypothesis_support": 2,
            },
            "overall_judgment": "ok",
            "confidence": 0.8,
        }
    )
    parsed = parse_rating_response(raw, "item_x")
    assert parsed["ratings"]["domain_meaning"] == 4


def test_parse_response_with_code_fence():
    raw = "```json\n" + json.dumps(
        {
            "item_id": "i",
            "ratings": {d: 3 for d in ["domain_meaning", "structural_plausibility", "readability_generalizability", "hypothesis_support"]},
        }
    ) + "\n```"
    parsed = parse_rating_response(raw, "i")
    assert all(v == 3 for v in parsed["ratings"].values())


def test_parse_rejects_out_of_range():
    raw = json.dumps(
        {"item_id": "i", "ratings": {"domain_meaning": 6, "structural_plausibility": 3, "readability_generalizability": 3, "hypothesis_support": 3}}
    )
    with pytest.raises(RatingValidationError):
        parse_rating_response(raw, "i")


def test_parse_rejects_missing_field():
    raw = json.dumps({"item_id": "i", "ratings": {"domain_meaning": 3}})
    with pytest.raises(RatingValidationError):
        parse_rating_response(raw, "i")


def test_parse_rejects_wrong_item_id():
    raw = json.dumps(
        {"item_id": "other", "ratings": {d: 3 for d in ["domain_meaning", "structural_plausibility", "readability_generalizability", "hypothesis_support"]}}
    )
    with pytest.raises(RatingValidationError):
        parse_rating_response(raw, "expected")


def test_parse_rejects_non_integer():
    raw = json.dumps(
        {"item_id": "i", "ratings": {"domain_meaning": 3.5, "structural_plausibility": 3, "readability_generalizability": 3, "hypothesis_support": 3}}
    )
    with pytest.raises(RatingValidationError):
        parse_rating_response(raw, "i")


def _fake_good_response(prompt: str) -> str:
    # extract item id from prompt
    import re

    m = re.search(r"item_id `([^`]+)`", prompt)
    item_id = m.group(1)
    return json.dumps(
        {
            "item_id": item_id,
            "ratings": {d: 4 for d in ["domain_meaning", "structural_plausibility", "readability_generalizability", "hypothesis_support"]},
            "overall_judgment": "fine",
            "confidence": 0.9,
        }
    )


def test_run_llm_judge_end_to_end_fake(tmp_path):
    export_blind_ratings(_candidates(), tmp_path / "m.jsonl", tmp_path / "p.json", seed=1)
    out = run_llm_judge(
        manifest_path=tmp_path / "m.jsonl",
        rubric_path=RUBRIC_PATH,
        out_dir=tmp_path / "judge",
        call_fn=_fake_good_response,
        model_id="fake_a",
        seed=1,
    )
    assert out["summary"]["n_rated"] == 2
    assert out["summary"]["n_errors"] == 0


def test_run_llm_judge_records_errors(tmp_path):
    export_blind_ratings(_candidates(), tmp_path / "m.jsonl", tmp_path / "p.json", seed=1)

    def bad(prompt):
        return "not json at all"

    out = run_llm_judge(
        manifest_path=tmp_path / "m.jsonl",
        rubric_path=RUBRIC_PATH,
        out_dir=tmp_path / "judge",
        call_fn=bad,
        model_id="fake_bad",
        seed=1,
        max_retries=2,
    )
    assert out["summary"]["n_errors"] == 2
    assert out["summary"]["n_rated"] == 0


def test_run_llm_judge_resume(tmp_path):
    export_blind_ratings(_candidates(), tmp_path / "m.jsonl", tmp_path / "p.json", seed=1)
    calls = {"n": 0}

    def counting(prompt):
        calls["n"] += 1
        return _fake_good_response(prompt)

    run_llm_judge(tmp_path / "m.jsonl", RUBRIC_PATH, tmp_path / "judge", counting, "fake_r", seed=1)
    first = calls["n"]
    # second run should hit cache and not call again
    run_llm_judge(tmp_path / "m.jsonl", RUBRIC_PATH, tmp_path / "judge", counting, "fake_r", seed=1)
    assert calls["n"] == first


def test_order_perturbation_differs_by_seed(tmp_path):
    # 3 items -> different seeds should generally give different presentation order
    export_blind_ratings(_candidates() + [
        {"formula": "v3", "variables": [], "method": "m", "dataset_label": "dataset_B"}
    ], tmp_path / "m.jsonl", tmp_path / "p.json", seed=1)
    o1 = run_llm_judge(tmp_path / "m.jsonl", RUBRIC_PATH, tmp_path / "j1", _fake_good_response, "a", seed=1)
    o2 = run_llm_judge(tmp_path / "m.jsonl", RUBRIC_PATH, tmp_path / "j2", _fake_good_response, "a", seed=999)
    pos1 = {r["item_id"]: r["presentation_position"] for r in o1["results"]}
    pos2 = {r["item_id"]: r["presentation_position"] for r in o2["results"]}
    assert pos1 != pos2


def test_score_interpretability_prompt_writes(tmp_path):
    export_blind_ratings(_candidates(), tmp_path / "m.jsonl", tmp_path / "p.json", seed=1)
    info = score_interpretability_prompt(tmp_path / "m.jsonl", RUBRIC_PATH, tmp_path / "prompts.json")
    assert info["n_prompts"] == 2
    payload = json.loads((tmp_path / "prompts.json").read_text())
    assert "v1 * v2" in payload[0]["prompt"] or "v1 + v2" in payload[0]["prompt"]
