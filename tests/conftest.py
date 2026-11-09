import os
import tempfile

import pytest

# Isolated vector store and deterministic agents for every test run.
os.environ["CHROMA_DIR"] = tempfile.mkdtemp(prefix="loanscout-chroma-")
os.environ["USE_LLM"] = "false"
os.environ.pop("LANGFUSE_PUBLIC_KEY", None)


@pytest.fixture
def scenarios():
    from scripts.scenarios import SCENARIOS
    return SCENARIOS
