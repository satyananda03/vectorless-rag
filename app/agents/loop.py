"""The PemDi audit agent: a deep agent reading the PageIndex library through its tools."""
from pathlib import Path

from deepagents import FilesystemMiddleware, create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.middleware import ToolLimitSynthesisMiddleware
from app.core.config import settings
from app.services.llm_service import get_llm
from app.services.prompt_service import load_prompt
from vectorless_rag import MultimodalClient
from vectorless_rag import pageindex_tools

SUPERAGENT_TOOL_LIMIT = 25
AGENT_ROOT = Path(__file__).resolve().parent
PROMPTS_PATH = AGENT_ROOT / "prompt.yaml"

SKILLS_SOURCE = "/skills"
FS_TOOLS = ["ls", "read_file", "glob", "grep"]
REGION = "Provinsi Jawa Timur"


def build_agent(model: BaseChatModel | None = None, storage_path: str | None = None):
    """Assemble the agent graph over the PageIndex library at ``storage_path`` (default: settings)."""
    prompt = load_prompt("agent", PROMPTS_PATH)["system_prompt"].format(region=REGION)
    library = MultimodalClient(storage_path=storage_path or settings.pageindex_storage_path)
    backend = FilesystemBackend(root_dir=AGENT_ROOT, virtual_mode=True)
    return create_deep_agent(
        model=model or get_llm(model_id=settings.pageindex_chat_model, streaming=True, max_tokens=11000),
        tools=pageindex_tools(library),
        system_prompt=prompt,
        backend=backend,
        skills=[],
        subagents=[],
        middleware=[
            FilesystemMiddleware(backend=backend, tools=FS_TOOLS),
            ToolLimitSynthesisMiddleware(run_limit=SUPERAGENT_TOOL_LIMIT),
        ],
        checkpointer=InMemorySaver(),
    )
