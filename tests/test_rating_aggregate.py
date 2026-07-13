from __future__ import annotations

import json

import pandas as pd

from factor_pysr_llm.rating_aggregate import (
    aggregate_ratings,
    inter_rater_agreement,
    llm_human_correlation,
    load_human_csv,
    load_llm_results,
    summarize_group,
)

DIMS = ["domain_meaning", "structural_plausibility", "readability_generalizability", "hypothesis_support"]


def _write_human_csv(path, rows):
    header = ["item_id", *DIMS, "rater_id"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_llm_jsonl(path, rows, model_id):
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            rec = {"item_id": r["item_id"], "model_id": model_id, "ratings": {d: r[d] for d in DIMS}}
            fh.write(json.dumps(rec) + "\n")


def test_summarize_group_means_and_ci(tmp_path):
    csv = tmp_path / "human.csv"
    _write_human_csv(csv, [
        {"item_id": "i1", "domain_meaning": 4, "structural_plausibility": 4, "readability_generalizability": 4, "hypothesis_support": 4, "rater_id": "h1"},
        {"item_id": "i2", "domain_meaning": 2, "structural_plausibility": 2, "readability_generalizability": 2, "hypothesis_support": 2, "rater_id": "h1"},
    ])
    df = load_human_csv(csv)
    summary = summarize_group(df, "human")
    assert summary["overall"]["mean"] == 3.0
    assert summary["dimensions"]["domain_meaning"]["mean"] == 3.0
    lo, hi = summary["overall"]["ci95"]
    assert lo <= 3.0 <= hi


def test_inter_rater_agreement(tmp_path):
    csv = tmp_path / "human.csv"
    _write_human_csv(csv, [
        {"item_id": "i1", "domain_meaning": 4, "structural_plausibility": 4, "readability_generalizability": 4, "hypothesis_support": 4, "rater_id": "h1"},
        {"item_id": "i2", "domain_meaning": 2, "structural_plausibility": 2, "readability_generalizability": 2, "hypothesis_support": 2, "rater_id": "h1"},
        {"item_id": "i1", "domain_meaning": 5, "structural_plausibility": 5, "readability_generalizability": 5, "hypothesis_support": 5, "rater_id": "h2"},
        {"item_id": "i2", "domain_meaning": 3, "structural_plausibility": 3, "readability_generalizability": 3, "hypothesis_support": 3, "rater_id": "h2"},
    ])
    df = load_human_csv(csv)
    agree = inter_rater_agreement(df)
    assert agree["n_raters"] == 2
    # perfectly rank-correlated raters
    assert agree["mean_pairwise_pearson"] > 0.99


def test_llm_human_correlation(tmp_path):
    human = tmp_path / "h.csv"
    _write_human_csv(human, [
        {"item_id": "i1", "domain_meaning": 5, "structural_plausibility": 5, "readability_generalizability": 5, "hypothesis_support": 5, "rater_id": "h1"},
        {"item_id": "i2", "domain_meaning": 3, "structural_plausibility": 3, "readability_generalizability": 3, "hypothesis_support": 3, "rater_id": "h1"},
        {"item_id": "i3", "domain_meaning": 1, "structural_plausibility": 1, "readability_generalizability": 1, "hypothesis_support": 1, "rater_id": "h1"},
    ])
    llm = tmp_path / "a.jsonl"
    _write_llm_jsonl(llm, [
        {"item_id": "i1", **{d: 4 for d in DIMS}},
        {"item_id": "i2", **{d: 3 for d in DIMS}},
        {"item_id": "i3", **{d: 2 for d in DIMS}},
    ], "llm_a")
    hdf = load_human_csv(human)
    ldf = load_llm_results(llm)
    corr = llm_human_correlation(hdf, ldf)
    assert corr["n_common_items"] == 3
    assert corr["spearman"] > 0.99
    assert corr["pairwise_agreement"] == 1.0


def test_aggregate_separates_human_and_llm(tmp_path):
    human = tmp_path / "h.csv"
    _write_human_csv(human, [
        {"item_id": "i1", "domain_meaning": 5, "structural_plausibility": 5, "readability_generalizability": 5, "hypothesis_support": 5, "rater_id": "h1"},
        {"item_id": "i2", "domain_meaning": 1, "structural_plausibility": 1, "readability_generalizability": 1, "hypothesis_support": 1, "rater_id": "h1"},
    ])
    la = tmp_path / "a.jsonl"
    lb = tmp_path / "b.jsonl"
    _write_llm_jsonl(la, [{"item_id": "i1", **{d: 4 for d in DIMS}}, {"item_id": "i2", **{d: 2 for d in DIMS}}], "llm_a")
    _write_llm_jsonl(lb, [{"item_id": "i1", **{d: 3 for d in DIMS}}, {"item_id": "i2", **{d: 3 for d in DIMS}}], "llm_b")
    report = aggregate_ratings(
        human_csv=human,
        llm_result_paths={"llm_a": la, "llm_b": lb},
        out_path=tmp_path / "report.json",
    )
    assert "human" in report["groups"]
    assert "llm_a" in report["groups"] and "llm_b" in report["groups"]
    # no combined human+llm overall number
    assert "combined" not in report["groups"]
    assert "human_vs_llm_a" in report["correlations"]
    assert (tmp_path / "report.json").exists()


def test_aggregate_with_private_map_unblind(tmp_path):
    human = tmp_path / "h.csv"
    _write_human_csv(human, [
        {"item_id": "i1", "domain_meaning": 5, "structural_plausibility": 5, "readability_generalizability": 5, "hypothesis_support": 5, "rater_id": "h1"},
        {"item_id": "i2", "domain_meaning": 2, "structural_plausibility": 2, "readability_generalizability": 2, "hypothesis_support": 2, "rater_id": "h1"},
    ])
    pmap = tmp_path / "private.json"
    pmap.write_text(json.dumps({"mapping": {
        "i1": {"method": "if_sr", "dataset": "D1"},
        "i2": {"method": "raw_pysr", "dataset": "D1"},
    }}), encoding="utf-8")
    report = aggregate_ratings(human_csv=human, llm_result_paths={}, private_map_path=pmap)
    assert report["unblinded"]["human_by_method"]["if_sr"]["overall_mean"] == 5.0
    assert report["unblinded"]["human_by_method"]["raw_pysr"]["overall_mean"] == 2.0
