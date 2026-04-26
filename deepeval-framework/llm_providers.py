"""
LLM provider abstraction layer with proper rate limiting.
Reads Groq rate limit headers and waits accordingly.
"""
import time
import threading
import requests as req
from config import OPENAI_API_KEY, GROK_API_KEY, GROQ_API_KEY, OLLAMA_BASE_URL, MODELS

# ─── Global rate limiter ───
_lock = threading.Lock()
_wait_until = 0  # timestamp when we can make the next call


def _enforce_rate_limit(resp_headers=None):
    """Wait based on rate limit headers or a minimum delay."""
    global _wait_until
    with _lock:
        now = time.time()
        if now < _wait_until:
            wait = _wait_until - now
            time.sleep(wait)

        # After a call, set next allowed time
        if resp_headers:
            remaining_tokens = int(resp_headers.get('x-ratelimit-remaining-tokens', 999))
            reset_tokens = resp_headers.get('x-ratelimit-reset-tokens', '1s')
            # Parse reset time like "1.5s" or "2m30s"
            reset_secs = _parse_duration(reset_tokens)
            if remaining_tokens < 2000:
                _wait_until = time.time() + max(reset_secs, 5)
            else:
                _wait_until = time.time() + 1.0
        else:
            _wait_until = time.time() + 3.0


def _parse_duration(s):
    """Parse Groq duration strings like '1.5s', '2m30s', '500ms'."""
    import re
    total = 0
    for val, unit in re.findall(r'([\d.]+)(ms|s|m|h)', str(s)):
        val = float(val)
        if unit == 'ms': total += val / 1000
        elif unit == 's': total += val
        elif unit == 'm': total += val * 60
        elif unit == 'h': total += val * 3600
    return total if total > 0 else 2.0


class LLMProvider:
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.model_config = MODELS[model_key]
        self.model_name = self.model_config["name"]

    def generate(self, prompt, system_prompt="", temperature=0.0, max_tokens=1024, json_mode=False):
        raise NotImplementedError


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, model_key, api_key):
        super().__init__(model_key)
        self.api_key = api_key
        self.base_url = self.model_config["base_url"]

    def generate(self, prompt, system_prompt="", temperature=0.0, max_tokens=1024, json_mode=False):
        _enforce_rate_limit()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = req.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=body,
                    timeout=60,
                )

                # Update rate limiter with response headers
                _enforce_rate_limit(resp.headers)

                if resp.status_code == 429:
                    retry_after = resp.headers.get('retry-after', None)
                    wait = float(retry_after) if retry_after else (attempt + 1) * 10
                    print(f"[provider] 429 rate limited, waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

            except req.exceptions.HTTPError as e:
                if "429" in str(e):
                    wait = (attempt + 1) * 10
                    print(f"[provider] Rate limited, waiting {wait}s")
                    time.sleep(wait)
                    continue
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                raise

        raise RuntimeError("Max retries exceeded")


class OllamaProvider(LLMProvider):
    def __init__(self, model_key="gemma"):
        super().__init__(model_key)

    def generate(self, prompt, system_prompt="", temperature=0.0, max_tokens=1024, json_mode=False):
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        resp = req.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": self.model_name, "prompt": full_prompt, "stream": False,
                  "options": {"temperature": temperature, "num_predict": max_tokens},
                  "format": "json" if json_mode else ""},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]


_GROQ_KEYS = {"groq", "groq_oss120b", "groq_scout", "groq_qwen"}


def get_provider(model_key):
    if model_key == "openai":
        return OpenAICompatibleProvider("openai", OPENAI_API_KEY)
    elif model_key == "grok":
        return OpenAICompatibleProvider("grok", GROK_API_KEY)
    elif model_key in _GROQ_KEYS:
        return OpenAICompatibleProvider(model_key, GROQ_API_KEY)
    elif model_key in ("oss_120b", "gemma"):
        return OllamaProvider(model_key)
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def get_judge():
    from config import JUDGE_LLM
    return get_provider(JUDGE_LLM)


def get_generator():
    from config import GENERATOR_LLM
    return get_provider(GENERATOR_LLM)
