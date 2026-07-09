from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def read_provider_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    provider = str(data.get("provider", "openai_compatible"))
    if provider != "openai_compatible":
        raise ValueError(f"unsupported provider: {provider}")
    return data


def _api_key(cfg: dict[str, Any]) -> str:
    explicit = str(cfg.get("api_key") or "").strip()
    if explicit:
        return explicit
    env_name = str(cfg.get("api_key_env") or "").strip()
    if env_name:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    raise RuntimeError(
        "No API key found. Set api_key in a local config file or set the env var named by api_key_env."
    )


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def call_openai_compatible(
    provider_config_path: Path,
    prompt: str,
    system_prompt: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = read_provider_config(provider_config_path)
    key = _api_key(cfg)
    messages = []
    sys_msg = system_prompt if system_prompt is not None else cfg.get("system_prompt")
    if sys_msg:
        messages.append({"role": "system", "content": str(sys_msg)})
    messages.append({"role": "user", "content": prompt})
    body: dict[str, Any] = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": float(cfg.get("temperature", 0.2)),
    }
    if cfg.get("max_tokens") is not None:
        body["max_tokens"] = int(cfg["max_tokens"])
    if response_format is not None:
        body["response_format"] = response_format
    elif cfg.get("response_format") is not None:
        body["response_format"] = cfg["response_format"]

    req = urllib.request.Request(
        _endpoint(str(cfg["base_url"])),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(cfg.get("timeout_seconds", 120))) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    return {
        "provider_config": str(provider_config_path),
        "model": cfg["model"],
        "content": content,
        "raw_response": data,
    }


def extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        json.loads(stripped)
        return stripped
    except Exception:
        pass
    starts = [i for i in (stripped.find("{"), stripped.find("[")) if i >= 0]
    if not starts:
        raise ValueError("model content does not contain a JSON object or array")
    start = min(starts)
    for end in range(len(stripped), start, -1):
        candidate = stripped[start:end].strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            continue
    raise ValueError("failed to extract valid JSON from model content")


def call_prompt_file(
    provider_config_path: Path,
    prompt_file: Path,
    output_path: Path,
    content_only: bool = False,
    extract_json: bool = False,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    prompt = prompt_file.read_text(encoding="utf-8")
    result = call_openai_compatible(provider_config_path, prompt, system_prompt=system_prompt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = str(result["content"])
    if extract_json:
        content = extract_json_text(content)
        output_path.write_text(content + "\n", encoding="utf-8")
    elif content_only:
        output_path.write_text(content, encoding="utf-8")
    else:
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "prompt_file": str(prompt_file),
        "output_path": str(output_path),
        "model": result["model"],
        "content_chars": len(str(result["content"])),
        "content_only": bool(content_only),
        "extract_json": bool(extract_json),
    }

