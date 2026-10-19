from collections.abc import Awaitable, Callable
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Diawali newline: content_blocks dirender berdempet, tanpa pemisah, sehingga tanpa ini kalimat ini menempel di akhir kalimat terakhir system prompt.
SYNTHESIS_PROMPT = (
    "\n\nYou have reached the maximum number of allowed tool calls for this loop. "
    "NEVER attempt to call any further tools. "
    "Using ONLY the tool results already present, provide your final answer now. Comperhensively"
)

#: Varian untuk agent ber-``response_format``: jawaban akhir memang berupa panggilan tool skema.
STRUCTURED_SYNTHESIS_PROMPT = (
    "\n\nYou have reached the maximum number of allowed search tool calls for this loop. "
    "Using ONLY the tool results already present, submit your final structured answer now."
)

class ToolLimitSynthesisMiddleware(AgentMiddleware):
    """Batasi eksekusi tool pada ``run_limit`` ToolMessage **per giliran**.

    Dihitung sejak HumanMessage terakhir, bukan seluruh thread: graf chat ber-checkpoint memuat
    riwayat semua giliran, dan hitungan kumulatif akan melarang tool selamanya setelah beberapa
    giliran — agent lalu dipaksa menjawab dari hasil tool lama yang mungkin sudah basi.

    Setelah batas tercapai, provider diminta menyintesis jawaban tanpa tool. Bila provider
    mengabaikan ``tool_choice="none"`` (mis. GLM-5 via Bedrock/LiteLLM), respons tool-call
    ilegalnya dibuang dan model dipanggil sekali lagi lewat ``_synthesis_request``: tanpa tool
    sama sekali, hasil tool dilebur ke pesan user — jadi tool ke-(T+1) tidak pernah dieksekusi.

    Tidak bergantung pada apa pun di project ini; cukup salin file ini ke agent lain.

    ``structured=True`` untuk agent dengan ``response_format`` (ToolStrategy): framework memaksa
    ``tool_choice="any"`` di setiap panggilan dan menambahkan tool skema SETELAH middleware, jadi
    ``tool_choice="none"`` diabaikan. Yang berlaku di sana adalah mengosongkan daftar tool biasa —
    satu-satunya tool yang tersisa adalah "jawab terstruktur", dan daftar tidak pernah kosong.
    """

    def __init__(self, run_limit: int, structured: bool = False) -> None:
        super().__init__()
        self.run_limit = run_limit
        self.structured = structured

    def _capped(self, request: ModelRequest) -> bool:
        used = 0
        for message in reversed(request.messages):
            if isinstance(message, HumanMessage):
                break
            used += isinstance(message, ToolMessage)
        return used >= self.run_limit

    def _apply(self, request: ModelRequest) -> ModelRequest:
        if not self._capped(request):
            return request

        # system_message bisa None (agent tanpa system prompt).
        blocks = list(request.system_message.content_blocks) if request.system_message else []
        if self.structured:
            return request.override(
                tools=[],
                system_message=SystemMessage(content=blocks + [{"type": "text", "text": STRUCTURED_SYNTHESIS_PROMPT}]),
            )
        return request.override(
            # tools=[] membuat payload kehilangan parameter `tools` sementara riwayatnya masih
            # berisi tool call → Bedrock via LiteLLM menolak dengan 400. Karena itu daftar
            # tool tetap dikirim, tetapi tool_choice="none" melarangnya pada provider patuh.
            # Provider yang mengabaikannya ditangani oleh _drop_illegal_tool_calls() di bawah.
            tool_choice="none",
            system_message=SystemMessage(
                content=blocks + [{"type": "text", "text": SYNTHESIS_PROMPT}]
            ),
        )

    def _synthesis_request(self, request: ModelRequest) -> ModelRequest:
        """Request sintesis tanpa tool, untuk provider yang mengabaikan ``tool_choice='none'``.

        ``tools=[]`` saja ditolak Bedrock (400) selama riwayat masih berisi tool call, jadi
        giliran ini diratakan: pertanyaan terakhir + seluruh hasil tool digabung jadi satu
        HumanMessage. Riwayat sebelum giliran ini dibiarkan apa adanya.
        """
        messages = request.messages
        cut = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), None)
        before, question, turn = (
            ([], "", messages) if cut is None else (messages[:cut], messages[cut].text, messages[cut + 1:])
        )
        results = "\n\n".join(f"[{m.name or 'tool'}]\n{m.text}" for m in turn if isinstance(m, ToolMessage))
        return self._apply(request).override(
            messages=[
                *before,
                HumanMessage(content=f"{question}\n\nTool results:\n\n{results}{SYNTHESIS_PROMPT}"),
            ],
            tools=[],
            tool_choice=None,
        )

    @staticmethod
    def _has_tool_calls(response: ModelResponse) -> bool:
        return any(isinstance(m, AIMessage) and m.tool_calls for m in response.result)

    def _drop_illegal_tool_calls(self, response: ModelResponse) -> ModelResponse:
        """Pengaman terakhir: buang ``tool_calls`` yang masih tersisa setelah sintesis ulang.

        Pertahankan teksnya bila ada, atau beri diagnostik eksplisit — tool ke-(T+1) tidak
        boleh dieksekusi dalam kondisi apa pun.
        """
        messages = []
        changed = False
        for message in response.result:
            if isinstance(message, AIMessage) and message.tool_calls:
                changed = True
                content = message.content or (
                    "Tool limit reached; provider attempted an additional tool call. "
                    "No additional tool was executed."
                )
                messages.append(message.model_copy(update={"content": content, "tool_calls": []}))
            else:
                messages.append(message)
        return ModelResponse(result=messages, structured_response=response.structured_response) if changed else response

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        response = handler(self._apply(request))
        if self._capped(request) and not self.structured and self._has_tool_calls(response):
            response = self._drop_illegal_tool_calls(handler(self._synthesis_request(request)))
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        # Wajib ada: AgentMiddleware.awrap_model_call bawaan melempar
        # NotImplementedError, dan seluruh jalur agentic_rag berjalan async.
        response = await handler(self._apply(request))
        if self._capped(request) and not self.structured and self._has_tool_calls(response):
            response = self._drop_illegal_tool_calls(await handler(self._synthesis_request(request)))
        return response

if __name__ == "__main__":
    # Self-check tanpa jaringan/DB: python -m src.core.middleware
    import asyncio
    from typing import Any

    from langchain.agents import create_agent
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool

    RUN_LIMIT = 3

    @tool
    def dummy_search(query: str) -> str:
        """Dummy search tool."""
        return f"hasil untuk {query}"

    class FakeModel(BaseChatModel):
        """Selalu memanggil tool selama ada tool. ``obeys_tool_choice=False`` meniru GLM-5:
        ``tool_choice="none"`` diabaikan."""

        bound_tools: list = []
        tool_choice: Any = None
        obeys_tool_choice: bool = True

        @property
        def _llm_type(self) -> str:
            return "fake"

        def bind_tools(self, tools, **kwargs) -> "FakeModel":
            clone = self.__class__(obeys_tool_choice=self.obeys_tool_choice)
            clone.bound_tools = list(tools or [])
            clone.tool_choice = kwargs.get("tool_choice")
            return clone

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            n = sum(1 for m in messages if isinstance(m, ToolMessage))
            blocked = self.obeys_tool_choice and self.tool_choice == "none"
            if self.bound_tools and not blocked:
                msg: Any = AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "dummy_search", "args": {"query": f"q{n}"}, "id": f"call_{n}"}
                    ],
                )
            else:
                msg = AIMessage(content="JAWABAN FINAL dari data yang ada")
            return ChatResult(generations=[ChatGeneration(message=msg)])

    async def main() -> None:
        for obeys in (True, False):
            agent = create_agent(
                model=FakeModel(obeys_tool_choice=obeys),
                tools=[dummy_search],
                system_prompt="You are a helper.",
                middleware=[ToolLimitSynthesisMiddleware(run_limit=RUN_LIMIT)],
            )
            result = await agent.ainvoke({"messages": [{"role": "user", "content": "halo"}]})
            messages = result["messages"]

            tool_messages = sum(1 for m in messages if isinstance(m, ToolMessage))
            assert tool_messages == RUN_LIMIT, f"expected {RUN_LIMIT} tool calls, got {tool_messages}"

            last = messages[-1]
            assert not getattr(last, "tool_calls", None), "final turn masih memanggil tool"
            assert "JAWABAN FINAL" in last.text, f"final message tidak sesuai: {last.content!r}"
            print(f"ok (obeys_tool_choice={obeys}) — berhenti tepat di {tool_messages} tool call, lalu sintesis")

    asyncio.run(main())