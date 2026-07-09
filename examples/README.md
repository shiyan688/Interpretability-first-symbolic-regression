# Examples

This directory contains a complete toy workflow that runs after cloning the repository.

## Files

- `toy_regression.csv`: small tabular regression dataset.
- `toy_config.json`: runnable workflow config. Paths are relative to this file.
- `factor_proposals.example.json`: example LLM-proposed factor JSON.
- `factor_selection.example.json`: example LLM factor-selection JSON.

## Run Without An LLM API

From the repository root:

```bash
python -m factor_pysr_llm.cli inspect-dataset --config examples/toy_config.json

python -m factor_pysr_llm.cli build-raw \
  --config examples/toy_config.json \
  --target y

python -m factor_pysr_llm.cli llm-propose-factors \
  --config examples/toy_config.json \
  --target y

python -m factor_pysr_llm.cli mine-factors \
  --config examples/toy_config.json \
  --target y \
  --llm-proposals examples/factor_proposals.example.json

python -m factor_pysr_llm.cli llm-select-factors \
  --config examples/toy_config.json \
  --target y

python -m factor_pysr_llm.cli build-pysr-pool \
  --config examples/toy_config.json \
  --target y \
  --llm-selection examples/factor_selection.example.json
```

Expected outputs:

```text
outputs/toy_regression_run/feature_tables/y/manifest.json
outputs/toy_regression_run/factor_pools/y/mined_factors.csv
outputs/toy_regression_run/llm_prompts/y/factor_proposal_prompt.md
outputs/toy_regression_run/feature_tables/y__pysr_pool/manifest.json
```

## Run With An LLM API

Create a private provider config:

```bash
cp configs/llm_provider.example.json configs/llm_provider.local.json
export OPENAI_API_KEY="your_api_key_here"
```

Generate a prompt:

```bash
python -m factor_pysr_llm.cli llm-propose-factors \
  --config examples/toy_config.json \
  --target y
```

Call the model and save JSON:

```bash
python -m factor_pysr_llm.cli llm-call \
  --provider-config configs/llm_provider.local.json \
  --prompt-file outputs/toy_regression_run/llm_prompts/y/factor_proposal_prompt.md \
  --output outputs/toy_regression_run/llm_prompts/y/factor_proposals.llm.json \
  --extract-json
```

Use the generated factors:

```bash
python -m factor_pysr_llm.cli mine-factors \
  --config examples/toy_config.json \
  --target y \
  --llm-proposals outputs/toy_regression_run/llm_prompts/y/factor_proposals.llm.json
```

