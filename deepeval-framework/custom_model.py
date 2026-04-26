"""
Custom DeepEval model wrapper with Groq JSON mode support.
"""
import json
import re
from deepeval.models import DeepEvalBaseLLM
from llm_providers import get_provider


def _schema_to_json_hint(schema_cls):
    if schema_cls is None:
        return ""
    try:
        schema = schema_cls.model_json_schema()
        props = schema.get("properties", {})
        example = {}
        for key, info in props.items():
            ptype = info.get("type", "string")
            if ptype == "array":
                items_info = info.get("items", {})
                if "properties" in items_info:
                    inner = {}
                    for ik, iv in items_info["properties"].items():
                        it = iv.get("type", "string")
                        inner[ik] = 0.5 if it in ("number", "integer") else (True if it == "boolean" else "example")
                    example[key] = [inner]
                else:
                    example[key] = ["example"]
            elif ptype in ("number", "integer"):
                example[key] = 0
            elif ptype == "boolean":
                example[key] = True
            else:
                example[key] = "example"
        return json.dumps(example, indent=2)
    except Exception:
        return ""


def _extract_json(text):
    """Extract JSON from response text."""
    # Try markdown code blocks
    m = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if m:
        text = m.group(1).strip()
    # Find JSON object/array
    for sc, ec in [('{', '}'), ('[', ']')]:
        si = text.find(sc)
        if si != -1:
            ei = text.rfind(ec)
            if ei > si:
                candidate = text[si:ei + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    fixed = re.sub(r',\s*([}\]])', r'\1', candidate)
                    try:
                        json.loads(fixed)
                        return fixed
                    except json.JSONDecodeError:
                        pass
    return text.strip()


class CustomEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_key="groq"):
        self.model_key = model_key
        self.provider = get_provider(model_key)
        self._model_name = self.provider.model_name
        super().__init__(model=self._model_name)

    def load_model(self):
        return self.provider

    def generate(self, prompt: str, schema=None, **kwargs) -> str:
        use_json = schema is not None

        if use_json:
            hint = _schema_to_json_hint(schema)
            prompt = prompt + f"\n\nYou MUST respond with ONLY valid JSON matching this structure:\n{hint}\nNo other text."
            sys_prompt = "You are a JSON evaluation assistant. Always respond with valid JSON only."
        else:
            sys_prompt = "You are a precise evaluation assistant."

        try:
            response = self.provider.generate(
                prompt,
                system_prompt=sys_prompt,
                temperature=0.0,
                max_tokens=2048,
                json_mode=use_json,
            )

            if use_json:
                cleaned = _extract_json(response)
                try:
                    parsed = json.loads(cleaned)
                    return json.dumps(parsed)
                except json.JSONDecodeError:
                    return cleaned

            return response

        except Exception as e:
            if use_json:
                return json.dumps({"error": str(e)})
            return f"Error: {str(e)}"

    async def a_generate(self, prompt: str, schema=None, **kwargs) -> str:
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return self._model_name
