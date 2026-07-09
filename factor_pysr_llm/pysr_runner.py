from __future__ import annotations

import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from .features import finite_frame, safe_read_csv


def configure_threads(procs: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("JULIA_NUM_THREADS", "1")
    os.environ["PYSR_NUM_THREADS"] = str(max(1, procs))
    os.environ["PYSR_USE_BEARTYPE"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"


def read_feature_dir(feature_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    x_path = feature_dir / "hybrid_features.csv"
    if not x_path.exists():
        x_path = feature_dir / "features.csv"
    y_path = feature_dir / "y.csv"
    X = finite_frame(safe_read_csv(x_path))
    y_df = safe_read_csv(y_path)
    y_col = "target" if "target" in y_df.columns else y_df.columns[0]
    y = pd.to_numeric(y_df[y_col], errors="coerce").to_numpy(dtype=float)
    return X, y


def run_pysr(
    feature_dir: Path,
    run_dir: Path,
    options: dict[str, Any],
    target: str,
) -> dict[str, Any]:
    from pysr import PySRRegressor

    procs = int(options.get("procs", 16))
    configure_threads(procs)
    run_dir.mkdir(parents=True, exist_ok=True)
    X, y = read_feature_dir(feature_dir)
    if X.shape[1] == 0:
        raise RuntimeError(f"no features in {feature_dir}")

    lin = LinearRegression().fit(X.to_numpy(dtype=float), y)
    lin_pred = lin.predict(X.to_numpy(dtype=float))
    linear_r2 = float(r2_score(y, lin_pred))
    linear_rmse = float(math.sqrt(mean_squared_error(y, lin_pred)))

    result: dict[str, Any] = {
        "target": target,
        "status": "started",
        "feature_dir": str(feature_dir),
        "run_dir": str(run_dir),
        "n_features": int(X.shape[1]),
        "linear_r2": linear_r2,
        "linear_rmse": linear_rmse,
        "options": options,
        "started_at_unix": time.time(),
    }
    (run_dir / "best_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(options, indent=2, ensure_ascii=False), encoding="utf-8")

    pysr_output_dir = Path(str(options.get("pysr_output_dir") or run_dir))
    pysr_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = PySRRegressor(
            model_selection=str(options.get("model_selection", "accuracy")),
            niterations=int(options.get("niterations", 1_000_000)),
            binary_operators=list(options.get("binary_operators", ["+", "-", "*", "/"])),
            unary_operators=list(options.get("unary_operators", ["abs", "inv(x)=1/x"])),
            extra_sympy_mappings={"inv": lambda x: 1 / x},
            elementwise_loss=str(options.get("elementwise_loss", "loss(x, y) = (x - y)^2")),
            population_size=int(options.get("population_size", 300)),
            populations=int(options.get("populations", 32)),
            maxsize=int(options.get("maxsize", 30)),
            parsimony=float(options.get("parsimony", 1.0e-6)),
            adaptive_parsimony_scaling=float(options.get("adaptive_parsimony_scaling", 150.0)),
            ncycles_per_iteration=int(options.get("ncycles_per_iteration", 250)),
            optimizer_algorithm=str(options.get("optimizer_algorithm", "BFGS")),
            optimizer_nrestarts=int(options.get("optimizer_nrestarts", 3)),
            optimizer_iterations=int(options.get("optimizer_iterations", 12)),
            should_optimize_constants=bool(options.get("should_optimize_constants", True)),
            optimize_probability=float(options.get("optimize_probability", 0.16)),
            tournament_selection_n=int(options.get("tournament_selection_n", max(2, min(15, int(options.get("population_size", 300)) - 1)))),
            precision=int(options.get("precision", 64)),
            procs=procs,
            parallelism=str(options.get("parallelism", "multiprocessing")),
            timeout_in_seconds=int(options.get("timeout_seconds", 24 * 3600)),
            random_state=int(options.get("seed", 20260709)),
            deterministic=bool(options.get("deterministic", False)),
            warm_start=bool(options.get("warm_start", False)),
            verbosity=int(options.get("verbosity", 0)),
            update_verbosity=int(options.get("update_verbosity", 0)),
            progress=bool(options.get("progress", False)),
            output_directory=str(pysr_output_dir),
        )
        t0 = time.time()
        model.fit(X, y)
        elapsed = time.time() - t0
        pred = np.asarray(model.predict(X), dtype=float)
        finite = np.isfinite(pred) & np.isfinite(y)
        best_r2 = float(r2_score(y[finite], pred[finite]))
        best_rmse = float(math.sqrt(mean_squared_error(y[finite], pred[finite])))
        equations = model.equations_
        equations.to_csv(run_dir / "model_equations_snapshot.csv", index=False)
        best_row = model.get_best()
        result.update(
            {
                "status": "success",
                "elapsed_seconds": elapsed,
                "best_r2": best_r2,
                "best_rmse": best_rmse,
                "best_equation": str(best_row.get("equation", "")),
                "best_complexity": int(best_row.get("complexity", -1)),
                "n_finite_pred": int(finite.sum()),
                "finished_at_unix": time.time(),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "finished_at_unix": time.time(),
            }
        )
        raise
    finally:
        (run_dir / "best_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

