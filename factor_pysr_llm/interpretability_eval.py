from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Interpretability evaluation infrastructure
# ------------------------------------------
# This module implements the blind rating pipeline required by the paper:
#   1. export a blinded rating manifest (jsonl) + isolated private mapping
#   2. build judge prompts that EMBED the formula, variable dictionary, rubric
#      and output schema (never a local file path)
#   3. strict schema validation of judge responses (no silent defaults)
#   4. an LLM judge runner with caching/resume/order-perturbation
#
# Method names, seeds and true R^2 are stored ONLY in the private mapping file,
# never in the rating manifest or the judge prompt.

RUBRIC_DIMENSIONS = (
    "domain_meaning",
    "structural_plausibility",
    "readability_generalizability",
    "hypothesis_support",
)


# ----------------------------------------------------------------------------
# Blind export
# ----------------------------------------------------------------------------
@dataclass
class RatingItem:
    item_id: str
    dataset_label: str
    formula: str
    variables: list[dict[str, Any]]
    task_context: str = ""

    def public_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "dataset_label": self.dataset_label,
            "formula": self.formula,
            "variables": self.variables,
            "task_context": self.task_context,
        }


_FORBIDDEN_PUBLIC_KEYS = {"method", "method_name", "seed", "r2", "r2_test", "r2_val", "explanation", "llm_explanation"}


def _anon_id(index: int, salt: str) -> str:
    h = hashlib.blake2b(f"{salt}:{index}".encode("utf-8"), digest_size=6).hexdigest()
    return f"item_{h}"


def export_blind_ratings(
    candidates: list[dict[str, Any]],
    out_manifest: Path,
    out_private_map: Path,
    seed: int = 20260709,
    dataset_label_field: str = "dataset_label",
) -> dict[str, Any]:
    """Export a randomized, anonymized rating manifest and an isolated private map.

    Each candidate must provide: formula, variables (list of {name,definition,
    unit,allowed_range}), and optional task_context. Private fields (method,
    seed, r2...) are moved into the private map only.
    """
    rng = random.Random(int(seed))
    order = list(range(len(candidates)))
    rng.shuffle(order)

    manifest_rows: list[dict[str, Any]] = []
    private_map: dict[str, Any] = {}
    for new_index, orig_index in enumerate(order):
        cand = candidates[orig_index]
        item_id = _anon_id(new_index, salt=str(seed))
        item = RatingItem(
            item_id=item_id,
            dataset_label=str(cand.get(dataset_label_field, cand.get("dataset_label", f"dataset_{new_index % 7}"))),
            formula=str(cand["formula"]),
            variables=list(cand.get("variables", [])),
            task_context=str(cand.get("task_context", "")),
        )
        public = item.public_dict()
        # defensive: ensure no forbidden key leaked into public payload
        leaked = _FORBIDDEN_PUBLIC_KEYS & set(public.keys())
        if leaked:
            raise ValueError(f"blind export would leak private keys: {sorted(leaked)}")
        manifest_rows.append(public)
        private_map[item_id] = {
            "orig_index": orig_index,
            "method": cand.get("method"),
            "seed": cand.get("seed"),
            "r2_test": cand.get("r2_test"),
            "r2_val": cand.get("r2_val"),
            "dataset": cand.get("dataset"),
            "target": cand.get("target"),
        }

    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_private_map = Path(out_private_map)
    out_private_map.parent.mkdir(parents=True, exist_ok=True)
    out_private_map.write_text(
        json.dumps({"seed": seed, "mapping": private_map}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # human rating CSV template (no scores yet)
    csv_path = out_manifest.with_suffix(".human_template.csv")
    header = ["item_id", *RUBRIC_DIMENSIONS, "overall_judgment", "rater_id"]
    lines = [",".join(header)]
    for row in manifest_rows:
        lines.append(",".join([row["item_id"], "", "", "", "", "", ""]))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "n_items": len(manifest_rows),
        "manifest_path": str(out_manifest),
        "private_map_path": str(out_private_map),
        "human_template_csv": str(csv_path),
        "seed": seed,
    }


def load_rating_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ----------------------------------------------------------------------------
# Judge prompt construction (content embedded, no file paths)
# ----------------------------------------------------------------------------
def load_rubric(path: Path) -> dict[str, Any]:
    rubric = json.loads(Path(path).read_text(encoding="utf-8"))
    dims = [d["key"] for d in rubric.get("dimensions", [])]
    if set(dims) != set(RUBRIC_DIMENSIONS):
        raise ValueError(f"rubric dimensions must be exactly {RUBRIC_DIMENSIONS}, got {dims}")
    return rubric


def _variables_block(variables: list[dict[str, Any]]) -> str:
    if not variables:
        return "(no variable dictionary provided)"
    lines = ["| variable | definition | unit | allowed range |", "| --- | --- | --- | --- |"]
    for v in variables:
        lines.append(
            f"| {v.get('name','')} | {v.get('definition','')} | {v.get('unit','')} | {v.get('allowed_range','')} |"
        )
    return "\n".join(lines)


def _rubric_block(rubric: dict[str, Any]) -> str:
    parts = []
    for dim in rubric["dimensions"]:
        anchors = "\n".join(f"      - {score}: {text}" for score, text in sorted(dim["anchors"].items()))
        parts.append(f"- **{dim['key']}** — {dim['title']}\n{anchors}")
    return "\n".join(parts)


def build_judge_prompt(item: dict[str, Any], rubric: dict[str, Any]) -> str:
    """Construct a self-contained judge prompt for one formula."""
    schema = json.dumps(rubric["output_schema"], ensure_ascii=False, indent=2)
    prompt = f"""# Formula Interpretability Rating

{rubric.get("instructions", "")}

## Task context

{item.get("task_context", "(none provided)")}

Dataset label: `{item.get("dataset_label", "")}`

## Variable dictionary

{_variables_block(item.get("variables", []))}

## Formula to rate

```
{item["formula"]}
```

## Rubric (1-5 each; use the anchors)

{_rubric_block(rubric)}

## Required output (strict JSON only)

Return exactly one JSON object matching this schema for item_id `{item["item_id"]}`:

```json
{schema}
```

Do not include any text outside the JSON object.
"""
    return prompt


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Strict response validation
# ----------------------------------------------------------------------------
class RatingValidationError(ValueError):
    pass


def _strip_code_fence(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


def parse_rating_response(raw: str, expected_item_id: str) -> dict[str, Any]:
    """Strictly parse and validate one judge response.

    Rejects: missing fields, out-of-range scores, non-integer scores, wrong or
    missing item_id. Never fills defaults silently.
    """
    text = _strip_code_fence(raw)
    try:
        data = json.loads(text)
    except Exception as exc:
        raise RatingValidationError(f"response is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise RatingValidationError("response JSON must be an object")
    item_id = data.get("item_id")
    if item_id is None:
        raise RatingValidationError("missing item_id")
    if str(item_id) != str(expected_item_id):
        raise RatingValidationError(f"item_id mismatch: got {item_id!r} expected {expected_item_id!r}")
    ratings = data.get("ratings")
    if not isinstance(ratings, dict):
        raise RatingValidationError("missing or invalid 'ratings' object")
    missing = set(RUBRIC_DIMENSIONS) - set(ratings.keys())
    if missing:
        raise RatingValidationError(f"missing rating dimensions: {sorted(missing)}")
    extra = set(ratings.keys()) - set(RUBRIC_DIMENSIONS)
    if extra:
        raise RatingValidationError(f"unexpected rating dimensions: {sorted(extra)}")
    clean_ratings: dict[str, int] = {}
    for dim in RUBRIC_DIMENSIONS:
        val = ratings[dim]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise RatingValidationError(f"{dim} must be a number, got {val!r}")
        if float(val) != int(val):
            raise RatingValidationError(f"{dim} must be an integer, got {val!r}")
        iv = int(val)
        if iv < 1 or iv > 5:
            raise RatingValidationError(f"{dim} out of range [1,5]: {iv}")
        clean_ratings[dim] = iv
    return {
        "item_id": str(item_id),
        "ratings": clean_ratings,
        "overall_judgment": str(data.get("overall_judgment", "")),
        "confidence": data.get("confidence"),
        "violation_flags": list(data.get("violation_flags", [])),
        "rationale": str(data.get("rationale", "")),
    }


# ----------------------------------------------------------------------------
# LLM judge runner (with caching, resume, retries, order perturbation)
# ----------------------------------------------------------------------------
def run_llm_judge(
    manifest_path: Path,
    rubric_path: Path,
    out_dir: Path,
    call_fn: Callable[[str], str],
    model_id: str,
    temperature: float = 0.0,
    seed: int = 20260709,
    max_retries: int = 3,
    batch_size: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run one LLM judge over all items.

    call_fn(prompt) -> raw string response. Injecting call_fn keeps this
    testable with fixed fake responses (no real API needed).

    Order is perturbed by ``seed`` so a second judge run can use a different
    seed to check position bias. Results are cached per item; on resume,
    already-parsed items are skipped.
    """
    items = load_rating_manifest(manifest_path)
    rubric = load_rubric(rubric_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"judge_{model_id}_seed{seed}.jsonl"
    errors_path = out_dir / f"judge_{model_id}_seed{seed}.errors.jsonl"

    done: dict[str, dict[str, Any]] = {}
    if resume and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["item_id"]] = rec

    # order perturbation
    rng = random.Random(int(seed))
    order = list(range(len(items)))
    rng.shuffle(order)
    if batch_size:
        # process in batches but order already shuffled globally
        pass

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    known_ids = {it["item_id"] for it in items}

    with cache_path.open("a", encoding="utf-8") as cache_fh, errors_path.open("a", encoding="utf-8") as err_fh:
        for pos, idx in enumerate(order):
            item = items[idx]
            item_id = item["item_id"]
            if item_id in seen_ids:
                raise RatingValidationError(f"duplicate item_id in manifest: {item_id}")
            seen_ids.add(item_id)
            if item_id in done:
                results.append(done[item_id])
                continue
            prompt = build_judge_prompt(item, rubric)
            phash = prompt_hash(prompt)
            last_err: str | None = None
            parsed: dict[str, Any] | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    raw = call_fn(prompt)
                    parsed = parse_rating_response(raw, item_id)
                    break
                except Exception as exc:  # includes RatingValidationError and API errors
                    last_err = f"{type(exc).__name__}: {exc}"
            if parsed is None:
                err = {
                    "item_id": item_id,
                    "presentation_position": pos,
                    "prompt_hash": phash,
                    "model_id": model_id,
                    "error": last_err,
                    "attempts": max_retries,
                }
                errors.append(err)
                err_fh.write(json.dumps(err, ensure_ascii=False) + "\n")
                continue
            record = {
                **parsed,
                "presentation_position": pos,
                "prompt_hash": phash,
                "model_id": model_id,
                "temperature": temperature,
                "seed": seed,
                "recorded_at_unix": time.time(),
            }
            results.append(record)
            cache_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # unknown item ids are impossible here (we iterate the manifest), but verify
    # missing coverage
    rated_ids = {r["item_id"] for r in results}
    missing_ids = known_ids - rated_ids
    summary = {
        "model_id": model_id,
        "seed": seed,
        "temperature": temperature,
        "n_items": len(items),
        "n_rated": len(results),
        "n_errors": len(errors),
        "n_missing": len(missing_ids),
        "missing_item_ids": sorted(missing_ids),
        "cache_path": str(cache_path),
        "errors_path": str(errors_path),
    }
    (out_dir / f"judge_{model_id}_seed{seed}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"results": results, "errors": errors, "summary": summary}


def score_interpretability_prompt(
    manifest_path: Path,
    rubric_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Write embedded judge prompts (one per item) for inspection / manual use."""
    items = load_rating_manifest(manifest_path)
    rubric = load_rubric(rubric_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for item in items:
        prompt = build_judge_prompt(item, rubric)
        payload.append({"item_id": item["item_id"], "prompt": prompt, "prompt_hash": prompt_hash(prompt)})
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"n_prompts": len(payload), "output_path": str(out_path)}
