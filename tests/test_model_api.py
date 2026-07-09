from __future__ import annotations

import json

from factor_pysr_llm.model_api import extract_json_text, read_provider_config


def test_extract_json_from_fenced_response() -> None:
    text = """Here is the result:

```json
{"selected_factors": [{"factor_name": "factor_000001"}]}
```
"""
    assert json.loads(extract_json_text(text))["selected_factors"][0]["factor_name"] == "factor_000001"


def test_read_provider_config(tmp_path) -> None:
    path = tmp_path / "provider.json"
    path.write_text(
        json.dumps(
            {
                "provider": "openai_compatible",
                "base_url": "https://example.com/v1",
                "api_key_env": "EXAMPLE_API_KEY",
                "model": "example-model",
            }
        ),
        encoding="utf-8",
    )
    cfg = read_provider_config(path)
    assert cfg["model"] == "example-model"

