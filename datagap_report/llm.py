from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
import logging
from typing import Any, Dict, Optional, Tuple

import httpx


class LLMClientError(Exception):
    pass


class LLMServerError(Exception):
    pass


class LLMClient:
    def __init__(
        self,
        api_base: str,
        api_key: Optional[str],
        cache_dir: Path,
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger("datagap_report.llm")
        # cost tracking
        self._cost = {
            "responses": {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0},
            "embeddings": {"calls": 0, "cache_hits": 0, "input_tokens": 0},
        }

    def _cache_key(self, model: str, payload: Dict[str, Any]) -> Path:
        body = json.dumps({"model": model, "payload": payload}, sort_keys=True).encode("utf-8")
        hexd = hashlib.sha256(body).hexdigest()
        return self.cache_dir / f"{hexd}.json"

    def _post(self, path: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.api_base}{path}"
        backoff = 1.0
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.post(url, headers=headers, json=json_body)
                if resp.status_code // 100 != 2:
                    rid = resp.headers.get("x-request-id")
                    body = resp.text
                    msg = f"HTTP {resp.status_code} {resp.reason_phrase}; request_id={rid}; body={body[:500]}"
                    if 400 <= resp.status_code < 500:
                        self.log.error(msg)
                        # bubble up client errors without retrying
                        raise LLMClientError(msg)
                    else:
                        self.log.warning(msg)
                        raise LLMServerError(msg)
                return resp.json()
            except LLMClientError as e:
                # no retry on client errors — re-raise to caller
                raise
            except Exception as e:
                last_err = e
                time.sleep(backoff)
                backoff = min(backoff * 2, 10)
        raise RuntimeError(f"request failed after retries: {last_err}")

    def responses_create(self, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Cache by model + payload
        key_path = self._cache_key(model, payload)
        if key_path.exists():
            self.log.debug(f"cache hit responses: {key_path.name}")
            self._cost["responses"]["cache_hits"] += 1
            return json.loads(key_path.read_text())
        self.log.info(f"LLM responses.create model={model} size={len(json.dumps(payload))}B")
        t0 = time.time()
        out = self._post("/v1/responses", {"model": model, **payload})
        self.log.info(f"LLM responses.create done in {time.time()-t0:.2f}s")
        self._cost["responses"]["calls"] += 1
        key_path.write_text(json.dumps(out))
        return out

    def structured_json(
        self,
        model: str,
        system: str,
        user: Dict[str, Any],
        response_format: Dict[str, Any],
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
        # Build format block in flattened shape expected by Responses API
        fmt = {
            "type": "json_schema",
            "name": response_format.get("name", "Structured"),
            "schema": response_format.get("schema", {}),
            "strict": response_format.get("strict", True),
        }
        payload: Dict[str, Any] = {
            "input": messages,
            # Structured Outputs moved to text.format
            "text": {"format": fmt},
        }
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        resp = self.responses_create(model=model, payload=payload)
        # Responses API returns output as JSON content; try to parse
        # Prefer top-level 'output' with JSON, else search in content
        data = None
        if isinstance(resp, dict):
            # heuristics to find JSON output
            if "output" in resp:
                data = resp["output"]
            elif "output_text" in resp:
                try:
                    data = json.loads(resp["output_text"])  # type: ignore
                except Exception:
                    data = None
            elif "content" in resp:
                try:
                    chunks = resp["content"]
                    if isinstance(chunks, list) and chunks:
                        # find first json
                        for ch in chunks:
                            if ch.get("type") == "output_text":
                                try:
                                    data = json.loads(ch.get("text", "{}"))
                                    break
                                except Exception:
                                    continue
                except Exception:
                    pass
        if data is None:
            raise RuntimeError("LLM response missing structured JSON")
        if not isinstance(data, dict):
            raise RuntimeError("LLM structured output is not an object")
        # estimate cost if usage missing
        usage = resp.get("usage") if isinstance(resp, dict) else None
        if isinstance(usage, dict):
            pt = int(usage.get("prompt_tokens", 0))
            ct = int(usage.get("completion_tokens", 0))
        else:
            # heuristic: chars/4 per role
            utext = messages[1]["content"]
            stext = messages[0]["content"]
            pt = int((len(utext) + len(stext)) / 4)
            ct = int((len(json.dumps(data)) / 4))
        self._cost["responses"]["prompt_tokens"] += pt
        self._cost["responses"]["completion_tokens"] += ct
        return data

    def embeddings(self, model: str, input_texts: list[str]) -> list[list[float]]:
        # Per-text cache: model + sha256(text)
        def text_sha(s: str) -> str:
            return hashlib.sha256(s.encode("utf-8")).hexdigest()

        cached: dict[int, list[float]] = {}
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        # probe cache
        for i, t in enumerate(input_texts):
            sha = text_sha(t)
            f = self.cache_dir / f"emb_{model}_{sha}.json"
            if f.exists():
                self._cost["embeddings"]["cache_hits"] += 1
                try:
                    cached[i] = json.loads(f.read_text())["embedding"]
                except Exception:
                    pass
            else:
                missing_indices.append(i)
                missing_texts.append(t)

        # batch missing in chunks up to 512
        batch_size = 512
        for start in range(0, len(missing_texts), batch_size):
            batch = missing_texts[start : start + batch_size]
            if not batch:
                continue
            payload = {"model": model, "input": batch}
            self.log.info(f"Embeddings create model={model} count={len(batch)}")
            t0 = time.time()
            resp = self._post("/v1/embeddings", payload)
            self.log.info(f"Embeddings done in {time.time()-t0:.2f}s")
            self._cost["embeddings"]["calls"] += 1
            # heuristic tokens count
            toks = sum(int(len(t) / 4) for t in batch)
            self._cost["embeddings"]["input_tokens"] += toks
            try:
                vecs = [row["embedding"] for row in resp["data"]]
            except Exception as e:
                raise RuntimeError(f"bad embeddings payload: {e}")
            # save per-text and place into cached map
            for off, vec in enumerate(vecs):
                idx = missing_indices[start + off]
                cached[idx] = vec
                sha = text_sha(input_texts[idx])
                (self.cache_dir / f"emb_{model}_{sha}.json").write_text(json.dumps({"embedding": vec}))

        # return in original order
        return [cached[i] if i in cached else [0.0] for i in range(len(input_texts))]

    def cost_summary(self) -> dict:
        return self._cost
