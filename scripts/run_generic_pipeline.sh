#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/generic_tabular_template.json}"
TARGET="${TARGET:-target_y}"
RUN_NAME="${RUN_NAME:-pysr_factor_pool_${TARGET}}"

"${PYTHON}" -m factor_pysr_llm.cli inspect-dataset \
  --config "${CONFIG}"

"${PYTHON}" -m factor_pysr_llm.cli build-raw \
  --config "${CONFIG}" \
  --target "${TARGET}"

"${PYTHON}" -m factor_pysr_llm.cli llm-propose-factors \
  --config "${CONFIG}" \
  --target "${TARGET}"

"${PYTHON}" -m factor_pysr_llm.cli mine-factors \
  --config "${CONFIG}" \
  --target "${TARGET}"

"${PYTHON}" -m factor_pysr_llm.cli llm-select-factors \
  --config "${CONFIG}" \
  --target "${TARGET}"

"${PYTHON}" -m factor_pysr_llm.cli build-pysr-pool \
  --config "${CONFIG}" \
  --target "${TARGET}"

"${PYTHON}" -m factor_pysr_llm.cli run-pysr \
  --config "${CONFIG}" \
  --target "${TARGET}" \
  --feature-dir "$(python - <<PY
import json
from pathlib import Path
cfg=json.loads(Path("${CONFIG}").read_text())
print(Path(cfg["output_root"]) / "feature_tables" / "${TARGET}__pysr_pool")
PY
)" \
  --run-name "${RUN_NAME}"

