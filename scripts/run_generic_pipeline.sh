#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/generic_tabular_template.json}"
TARGET="${TARGET:-y}"
RUN_NAME="${RUN_NAME:-pysr_factor_pool_${TARGET}}"
LLM_PROPOSALS="${LLM_PROPOSALS:-examples/factor_proposals.example.json}"
LLM_SELECTION="${LLM_SELECTION:-examples/factor_selection.example.json}"
RUN_PYSR="${RUN_PYSR:-0}"

"${PYTHON}" -m factor_pysr_llm.cli inspect-dataset \
  --config "${CONFIG}"

"${PYTHON}" -m factor_pysr_llm.cli build-raw \
  --config "${CONFIG}" \
  --target "${TARGET}"

"${PYTHON}" -m factor_pysr_llm.cli llm-propose-factors \
  --config "${CONFIG}" \
  --target "${TARGET}"

mine_args=(
  -m factor_pysr_llm.cli mine-factors
  --config "${CONFIG}"
  --target "${TARGET}"
)
if [[ -f "${LLM_PROPOSALS}" ]]; then
  mine_args+=(--llm-proposals "${LLM_PROPOSALS}")
fi
"${PYTHON}" "${mine_args[@]}"

"${PYTHON}" -m factor_pysr_llm.cli llm-select-factors \
  --config "${CONFIG}" \
  --target "${TARGET}"

pool_args=(
  -m factor_pysr_llm.cli build-pysr-pool
  --config "${CONFIG}"
  --target "${TARGET}"
)
if [[ -f "${LLM_SELECTION}" ]]; then
  pool_args+=(--llm-selection "${LLM_SELECTION}")
fi
"${PYTHON}" "${pool_args[@]}"

if [[ "${RUN_PYSR}" == "1" ]]; then
  "${PYTHON}" -m factor_pysr_llm.cli run-pysr \
    --config "${CONFIG}" \
    --target "${TARGET}" \
    --run-name "${RUN_NAME}"
else
  echo "Prepared PySR feature pool. Set RUN_PYSR=1 to launch PySR."
fi
