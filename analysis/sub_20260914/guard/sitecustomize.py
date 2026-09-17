"""Neutralize hosted APIs: blank credentials, unroute endpoints.

Imported automatically by CPython at interpreter start when this directory is on
sys.path. The campaign contract is zero paid API calls and the union protocol
sets allow_hosted_api=false with no hosted-judge fallback. Importing a client
library is harmless; calling one is not, so this removes the credential and the
endpoint rather than the import.
"""
import os
import sys

_CRED = ("OPENAI", "ANTHROPIC", "AZURE_OPENAI", "COHERE", "TOGETHER",
         "REPLICATE", "MISTRAL", "DEEPSEEK", "GOOGLE_API", "GEMINI")
for _name in [k for k in os.environ if any(s in k.upper() for s in _CRED)]:
    os.environ[_name] = ""

# an unroutable endpoint: port 1 on loopback refuses immediately
_DEAD = "http://127.0.0.1:1"
for _var in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_BASE_URL",
             "AZURE_OPENAI_ENDPOINT", "COHERE_API_URL", "TOGETHER_BASE_URL"):
    os.environ[_var] = _DEAD
os.environ["ALLOW_HOSTED_API"] = "0"


class _RefuseHostedHarness:
    """Refuse only the harness whose purpose is to orchestrate paid judging."""

    BLOCKED = ("alpaca_eval",)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ImportError(
                "%s is blocked: it orchestrates hosted judging, and this campaign "
                "makes zero paid API calls. Use the local evaluator." % fullname)
        return None


sys.meta_path.insert(0, _RefuseHostedHarness())
