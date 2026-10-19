"""PageIndex SDK client: the 0.2.x cloud surface, now with a local mode."""
from __future__ import annotations

import os
import re
import threading
import time
import warnings
from typing import (Any, Callable, Mapping,
                    Optional, Union, cast)

from .errors import PageIndexAPIError


_litellm_preload_started = False


def _preload_litellm() -> None:
    """Start litellm's multi-second import in the background, once per
    process — a per-client thread would churn under per-request clients."""
    global _litellm_preload_started
    if _litellm_preload_started:
        return
    _litellm_preload_started = True

    def _import() -> None:
        try:
            import litellm  # noqa: F401
        except Exception:
            pass

    threading.Thread(target=_import, daemon=True).start()


def _parse_pages(pages: str) -> list[int]:
    from .agent_tools import _PageSpecError, _expand_pages
    if isinstance(pages, str):
        # 0.2.10 tolerated whitespace on this surface; the tool layer stays
        # on the strict contract pattern.
        pages = re.sub(r"\s*([,-])\s*", r"\1", pages.strip())
    try:
        return _expand_pages(pages)
    except _PageSpecError as exc:
        raise PageIndexAPIError(str(exc)) from exc


# The two citation tag formats PageIndex chat writes and renders.
_OLD_CITATION_RE = re.compile(
    r"<doc=([^;<>]+);page=(\d+)(?:;block(?:_id)?=([^;<>]+))?>")
_CITE_TAG_RE = re.compile(r"<cite\s([^<>]*)>(?:(?P<inner>[^<>]*)</cite>)?")
_CITE_ATTR_RE = re.compile(r"""\b(\w+)=(["'])(.*?)\2""", re.S)


def _citation_key(m: re.Match) -> Optional[tuple[str, int, Optional[str]]]:
    """(document, page, block_id) of one matched tag, or None when it names
    no document or no positive page."""
    if m.re is _OLD_CITATION_RE:
        doc, page_str, block_id = m.group(1), m.group(2), m.group(3)
    else:
        attrs = {name: value for name, _, value in
                 _CITE_ATTR_RE.findall(m.group(1))}
        doc, page_str, block_id = (attrs.get("doc", ""), attrs.get("page", ""),
                                   attrs.get("block"))
    doc = doc.strip()
    block_id = (block_id or "").strip() or None
    try:
        page = int(page_str.split("-")[0])
    except ValueError:
        return None
    return (doc, page, block_id) if doc and page > 0 else None


def _parse_citations(text: str) -> list[dict[str, Any]]:
    """``<doc=…;page=…;block=…>`` tags (the managed chat's format), then
    ``<cite doc= page= block=/>`` tags; deduplicated, ``block_id`` only
    when the tag carries one."""
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, int, Optional[str]]] = set()
    for m in [*_OLD_CITATION_RE.finditer(text), *_CITE_TAG_RE.finditer(text)]:
        key = _citation_key(m)
        if key and key not in seen:
            seen.add(key)
            entry: dict[str, Any] = {"document": key[0], "page": key[1]}
            if key[2]:
                entry["block_id"] = key[2]
            found.append(entry)
    return found


def _yaml_names_chat(loader) -> bool:
    # config.yaml is a third way to name a chat model. Blank values mean
    # "absent", exactly like the flat arguments (_resolve_models agrees).
    return any(loader._default_dict.get(key)
               for key in ("chat_model", "retrieve_model", "model"))


_LOCAL_INDEX_KEYS = ("model", "summary_model", "backend", "storage_path",
                     "summary_max_words", "summary_concurrency",
                     "summary_reasoning_effort", "use_embedded_toc", "optimize")

# Near-synonyms of "cloud" that would otherwise parse as model names —
# a silent wrong mode. They error, pointing at the real word.
_RESERVED_MODE_WORDS = {"hosted", "managed"}


def _env_cloud_key(spelling: str, inline: str = "api_key=...") -> str:
    # .env support lives in utils' import-time load_dotenv(): load it
    # before the read, or a key in .env is visible only by import order.
    from . import utils  # noqa: F401
    key = os.environ.get("PAGEINDEX_API_KEY")
    if not key:
        raise PageIndexAPIError(
            f"{spelling} reads the PageIndex API key from the "
            "PAGEINDEX_API_KEY environment variable, which is not set — "
            f"export it, or pass the key inline ({inline}).")
    return key


# One argument vocabulary regardless of spelling: these values are shape-
# checked in the constructor, so a wrong type or an empty value refuses
# there as a PageIndexAPIError — never later, never silently.
_ARG_TYPES: "dict[str, tuple[type, ...]]" = {
    "model": (str,), "index_model": (str,), "summary_model": (str,),
    "chat_model": (str,), "retrieve_model": (str,), "summary_max_words": (int,),
    "summary_concurrency": (int,), "summary_reasoning_effort": (str,),
    "use_embedded_toc": (bool,), "optimize": (str,),
    "storage_path": (str, os.PathLike), "index_backend": (dict,),
    "chat_backend": (dict,)}


def _declared_mode(value, side: str):
    if isinstance(value, str):
        value = value.strip().lower()
    if value not in (None, "cloud", "local"):
        raise PageIndexAPIError(
            f'{side} "mode" must be "cloud" or "local", not {value!r}.')
    return value


_CloudKey = Union[str, Callable[[], str], None]


def _resolve_index_slot(index) -> "tuple[_CloudKey, dict[str, Any]]":
    """The ``index=`` slot as (cloud api_key, local overrides). A dict
    declares its side by its keys; an optional "mode" states it and must
    agree. Keyless cloud spellings ("cloud" / "pageindex-cloud",
    {"mode": "cloud"}) return the environment read as a thunk, so the
    caller's mode cross-check runs before the environment is touched."""
    from .types import PAGEINDEX_CLOUD
    if isinstance(index, str):
        # Normalized compare: a case/whitespace variant of a mode word
        # must never fall through and silently become a model name.
        word = index.strip().lower()
        if word in (PAGEINDEX_CLOUD, "cloud"):
            return lambda: _env_cloud_key(f'index="{index.strip()}"',
                                          'index={"api_key": ...}'), {}
        if word == "local":
            return None, {}
        if word in _RESERVED_MODE_WORDS:
            raise PageIndexAPIError(
                f'index="{index}" is not a mode word — the cloud spelling '
                'is index="cloud" (key from PAGEINDEX_API_KEY) or '
                'index={"api_key": ...}.')
        if index.strip():
            return None, {"index_model": index}
        raise PageIndexAPIError(
            "index is an empty string — pass a local index model name, "
            'or "cloud".')
    if isinstance(index, Mapping):
        # None-valued keys mean "absent", exactly like the flat arguments.
        conf: dict[str, Any] = {name: value for name, value in index.items()
                                if value is not None}
        declared = _declared_mode(conf.pop("mode", None), "index")
        if not conf:
            if declared == "cloud":
                return lambda: _env_cloud_key('index={"mode": "cloud"}',
                                              'index={"api_key": ...}'), {}
            if declared == "local":
                return None, {}
            raise PageIndexAPIError(
                "index is an empty dict — its keys pick the side: "
                '{"api_key": ...} for cloud documents, or '
                f"{', '.join(_LOCAL_INDEX_KEYS)} for the local store.")
        unknown = set(conf) - {"api_key"} - set(_LOCAL_INDEX_KEYS)
        if unknown:
            raise PageIndexAPIError(
                f"Unknown index keys ({', '.join(sorted(unknown))}) — "
                'cloud takes "api_key"; local takes '
                f"{', '.join(_LOCAL_INDEX_KEYS)}.")
        if "api_key" in conf:
            if declared == "local":
                raise PageIndexAPIError(
                    'index declares mode "local" but carries api_key — '
                    "an API key means cloud documents. Drop one of them.")
            if len(conf) > 1:
                raise PageIndexAPIError(
                    "index mixes cloud and local keys — cloud documents "
                    'take {"api_key": ...} alone; the cloud pipeline does '
                    "its own indexing.")
            key = conf["api_key"]
            if not key or not isinstance(key, str):
                raise PageIndexAPIError(
                    'index["api_key"] must be a non-empty string.')
            return key, {}
        if declared == "cloud":
            raise PageIndexAPIError(
                'index declares mode "cloud" but carries local keys '
                f"({', '.join(sorted(conf))}) — the cloud pipeline does "
                'its own indexing; cloud takes "api_key" only.')
        rename = {"model": "index_model", "backend": "index_backend"}
        return None, {rename.get(name, name): value for name, value in conf.items()}
    raise PageIndexAPIError("index must be a string or a dict.")


def _resolve_chat_slot(chat) -> "tuple[Optional[str], dict[str, Any]]":
    """The ``chat=`` slot as (mode, own-model overrides) — mode is
    "managed", "own", or None (nothing declared beyond the overrides)."""
    from .types import PAGEINDEX_CLOUD
    if isinstance(chat, str):
        word = chat.strip().lower()
        if word in (PAGEINDEX_CLOUD, "cloud"):
            return "managed", {}
        if word == "local":
            return "own", {}
        if word in _RESERVED_MODE_WORDS:
            raise PageIndexAPIError(
                f'chat="{chat}" is not a mode word — the managed chat is '
                'chat="cloud".')
        if chat.strip():
            return "own", {"chat_model": chat}
        raise PageIndexAPIError(
            "chat is an empty string — pass a model name, or "
            '"cloud" for the managed chat.')
    if isinstance(chat, Mapping):
        # None-valued keys mean "absent", exactly like the flat arguments.
        conf = {name: value for name, value in chat.items()
                if value is not None}
        declared = _declared_mode(conf.pop("mode", None), "chat")
        unknown = set(conf) - {"model", "backend"}
        if (not conf and declared is None) or unknown:
            raise PageIndexAPIError(
                ("chat is an empty dict" if not conf else
                 f"Unknown chat keys ({', '.join(sorted(unknown))})")
                + ' — chat takes "model" and "backend" (your own model), '
                'or {"mode": "cloud"} / "cloud" for the managed chat.')
        if declared == "cloud":
            if conf:
                raise PageIndexAPIError(
                    'chat declares mode "cloud" but carries '
                    f"({', '.join(sorted(conf))}) — the managed chat "
                    "selects its own model. Drop the mode, or the keys.")
            return "managed", {}
        mapped = {"chat_model": conf.get("model"),
                  "chat_backend": conf.get("backend")}
        return "own", {name: value for name, value in mapped.items()
                       if value is not None}
    raise PageIndexAPIError("chat must be a string or a dict.")


class PageIndexClient:
    """
    Python SDK client for PageIndex.

    Two independent sides, each locally run or cloud-managed:

    - **index** — where documents live. With an ``api_key`` they live in
      your PageIndex cloud account, indexed by the managed pipeline,
      exactly like the 0.2.x SDK. Without one they are indexed on your
      machine by the open-source pipeline (your own LLM provider key,
      e.g. ``OPENAI_API_KEY``) and stored under ``storage_path``.
    - **chat** — who answers. With a chat model configured
      (``chat_model=`` / ``chat=``), the document-QA agent runs in your
      process against your own model and credentials — in both index
      modes. On a cloud client with no chat model, the managed cloud
      chat answers.

    ``api_key`` moves your documents, never your model: ``chat_model``
    always means your own model on your own keys. The fourth combination
    (local documents + managed chat) cannot be expressed — the managed
    chat cannot read your disk.

    Usage:
        client = PageIndexClient()                  # local docs + your model
        client = PageIndexClient(api_key="...")     # cloud docs + managed chat
        client = PageIndexClient(api_key="...",     # cloud docs + your model
                                 chat_model="openai/gpt-6-sol")

    ``index=`` / ``chat=`` are the grouped spelling of the same flat
    arguments — a string as shorthand, a dict for the full config; each
    side picks one spelling per client. ``index="cloud"`` (or the label
    ``"pageindex-cloud"``) is the keyless cloud spelling (the key comes
    from the ``PAGEINDEX_API_KEY`` environment variable — which is read
    only when the code explicitly says cloud; a bare ``PageIndexClient()``
    stays local regardless of the environment).

    Args:
        api_key (str, optional): PageIndex cloud API key
            (https://developer.pageindex.ai/api-keys). Omit for local mode.
        index (str | dict, optional): The index side, grouped —
            ``"cloud"`` / ``"pageindex-cloud"`` (cloud, key from the
            environment), ``"local"``, a local index model name, or a
            dict: ``{"api_key": ...}`` for cloud, ``{"model",
            "summary_model", "backend", "storage_path",
            "summary_max_words", "summary_concurrency",
            "summary_reasoning_effort", "use_embedded_toc", "optimize"}``
            for local. An
            optional ``"mode"`` key (``"cloud"`` / ``"local"``) states
            the side and must agree with the other keys; ``{"mode":
            "cloud"}`` alone reads the key from the environment. Not
            combinable with this side's flat arguments.
        chat (str | dict, optional): The chat side, grouped — a model
            name (your own model), ``"cloud"`` / ``"pageindex-cloud"``
            (managed chat, cloud clients only), ``"local"`` (your own
            model, the default one), or ``{"model", "backend"}``. An
            optional ``"mode"`` key states the side: ``{"mode":
            "cloud"}`` alone is the managed chat, ``"local"`` is your
            own model and must agree with the other keys. Not
            combinable with this side's flat arguments.
        mode (str, optional): Client-level declaration of where the
            documents live — ``"cloud"`` or ``"local"`` — checked
            against the other arguments (``mode="local"`` with an
            api_key errors). ``mode="cloud"`` alone reads the key from
            the ``PAGEINDEX_API_KEY`` environment variable. Always
            optional: the arguments themselves already carry the mode.
        index_model (str, optional): Local mode only — LLM used to index
            documents (structure and summaries). Defaults to the SDK
            default (fast and cheap).
        chat_model (str, optional): Your own model for the chat surfaces
            (``chat``, ``chat_completions``; a value you set also
            carries onto ``chat(protocol="messages")`` and the two
            Anthropic agent configs), exposed as ``client.chat_model`` —
            on a cloud client, setting it runs the document-QA agent in
            your process over the cloud documents (page content then
            flows through your process to your model provider). Chat
            names route through LiteLLM and mean what LiteLLM says they
            mean; bare names are OpenAI-compatible shorthand, and
            ``openai/Qwen/...`` is the form for an OpenAI-compatible
            server that itself serves slashed model ids (vLLM, TGI).
            ``chat(protocol="responses")`` and the Anthropic-native
            surfaces read the name by its routing prefix instead and
            send the rest as written — bare names are OpenAI's and
            Anthropic's own respectively — and the Anthropic-native ones
            treat the untouched stock default as no choice. Defaults to the
            SDK default (strong); reads ``None`` on a cloud client where
            the managed chat answers.
        model (str, optional): Local mode only — one model for both roles:
            sets the default for ``index_model`` and ``chat_model`` at
            once. The role-specific arguments win over it. (Also the
            0.2.8-era name for the indexing model — old configs keep
            working unchanged.)
        summary_model (str, optional): Local mode only — legacy: overrides
            the model used for node summaries and document descriptions;
            ``index_model`` covers this.
        summary_max_words (int, optional): Local flash mode only — the word
            cap each model-written node summary is asked to stay within;
            a leaf node shorter than that keeps its own text. Defaults to 150.
        summary_concurrency (int, optional): Local flash mode only — cap on
            simultaneous indexing model calls per lane: the summaries, and
            expand up to its own ceiling of 32. The lanes overlap, so up to
            cap + min(32, cap) calls run at once. Defaults to 64.
        summary_reasoning_effort (str, optional): Local flash mode only —
            the reasoning effort sent with each node-summary call, over any
            in ``index_backend`` (``"none"``, ``"low"``, ``"high"``, ...; a
            model without the setting drops it). Defaults to ``"none"``
            unless LiteLLM lists the model as refusing it, and stops
            sending it once a prompt the server refused with it (HTTP
            400) goes through without it.
            Expand keeps the model's own default. A
            ``mode="standard"`` submit refuses any of the summary knobs.
        use_embedded_toc (bool, optional): Local mode only — whether flash
            indexing consumes the PDF's embedded bookmarks when they look
            trustworthy. Defaults to True.
        optimize (str, optional): Local mode only — the flash tree
            refinement pass: ``"full"`` (merge + model expand, the
            default), ``"merge"`` (deterministic merge only) or ``"off"``.
        retrieve_model (str, optional): Legacy name for ``chat_model`` —
            same meaning everywhere, cloud clients included.
        storage_path (str or os.PathLike, optional): Local mode only —
            directory where indexed documents are stored. Defaults to
            ``./.pageindex``.
        index_backend (dict, optional): Local mode only — connection
            overrides for the indexing lane's LLM calls. Keys are
            LiteLLM's own connection params — ``api_key``, ``api_base``,
            ``api_version``, ``aws_*``, … — passed through verbatim.
        chat_backend (dict, optional): Default connection overrides for
            the chat surfaces — a chat-side argument like ``chat_model``,
            so on a cloud client it selects own-model chat. A call's own
            ``backend`` keys win over it. The dict reaches whichever
            door runs, in that door's vocabulary (see each method) —
            ``api_key`` / ``base_url`` mean the same thing on all three.
        instructions (str, optional): Standing guidance for the answering
            agent — persona, language, format — appended after the
            managed system prompt on every chat surface, the managed
            cloud chat included, and in ``agent_instructions()`` and the
            ``*_agent_config()`` bundles. Not a chat-side spelling: it
            combines with any ``chat=`` and never selects own-model chat.
            ``chat(instructions=...)`` adds to it per call. Indexing
            has no prompt to extend.

    PageIndexCloudClient / PageIndexLocalClient pin the index side at
    construction instead of inferring it from api_key.

    Local mode differences (all documented per method): indexing is
    synchronous, only PDFs are supported, and folders / ``beta_headers`` /
    the deprecated retrieval API (``submit_query``, ``get_retrieval``) are
    cloud-only.
    """

    BASE_URL = "https://api.pageindex.ai"

    _pin: Optional[str] = None  # the pinned subclasses' index side

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        index: Optional[Union[Mapping[str, Any], str]] = None,
        chat: Optional[Union[Mapping[str, Any], str]] = None,
        mode: Optional[str] = None,
        index_model: Optional[str] = None,
        chat_model: Optional[str] = None,
        model: Optional[str] = None,
        summary_model: Optional[str] = None,
        summary_max_words: Optional[int] = None,
        summary_concurrency: Optional[int] = None,
        summary_reasoning_effort: Optional[str] = None,
        use_embedded_toc: Optional[bool] = None,
        optimize: Optional[str] = None,
        retrieve_model: Optional[str] = None,
        storage_path: Optional[Union[str, os.PathLike[str]]] = None,
        index_backend: Optional[dict[str, Any]] = None,
        chat_backend: Optional[dict[str, Any]] = None,
        instructions: Optional[str] = None,
    ):
        if api_key == "":
            raise PageIndexAPIError(
                "api_key is an empty string. Pass a real PageIndex API key for "
                "cloud mode, or omit api_key entirely for local mode."
            )
        if instructions is not None and not isinstance(instructions, str):
            raise PageIndexAPIError(
                f"instructions must be a str, got {type(instructions).__name__}. "
                "Pass the guidance as text; Messages system blocks belong to "
                "chat(protocol=\"messages\", model=..., instructions=[...]) on "
                "a client with chat_model=... set.")
        self.instructions = (instructions or "").strip() or None
        # Each side picks one spelling — its slot, or the flat arguments.
        # ``model`` sets every role, so it claims both sides.
        index_flat: dict[str, Any] = {
            name: value for name, value in
            (("api_key", api_key),
             ("index_model", index_model),
             ("summary_model", summary_model),
             ("summary_max_words", summary_max_words),
             ("summary_concurrency", summary_concurrency),
             ("summary_reasoning_effort", summary_reasoning_effort),
             ("use_embedded_toc", use_embedded_toc),
             ("optimize", optimize),
             ("index_backend", index_backend),
             ("storage_path", storage_path), ("model", model))
            if value is not None}
        chat_flat: dict[str, Any] = {
            name: value for name, value in
            (("chat_model", chat_model),
             ("retrieve_model", retrieve_model),
             ("chat_backend", chat_backend), ("model", model))
            if value is not None}
        if model is not None and (index is not None or chat is not None):
            raise PageIndexAPIError(
                "model= sets both roles at once, so no slot can absorb "
                'it — name the model inside the slot ({"model": ...}) '
                "and use index_model= / chat_model= for a side written "
                "flat.")
        if index is not None and index_flat:
            raise PageIndexAPIError(
                "index= and the flat index-side arguments "
                f"({', '.join(sorted(index_flat))}) are two spellings of "
                "the same thing — use one or the other.")
        if chat is not None and chat_flat:
            raise PageIndexAPIError(
                "chat= and the flat chat-side arguments "
                f"({', '.join(sorted(chat_flat))}) are two spellings of "
                "the same thing — use one or the other.")
        # ``mode=`` is a cross-check, not a spelling: it combines with
        # either spelling of the index side and must agree with it. The
        # pinned classes declare the side by class; their errors name the
        # class, never a mode= the user did not write.
        declared = _declared_mode(mode, "client") or self._pin
        pinned = type(self).__name__ if self._pin else None
        if index is not None:
            cloud_key, index_conf = _resolve_index_slot(index)
            if declared == "local" and cloud_key is not None:
                raise PageIndexAPIError(
                    f"{pinned} pins local documents — that index= selects "
                    "cloud documents. Drop it, or use PageIndexCloudClient."
                    if pinned else
                    'mode="local" disagrees with index= — that index '
                    "selects cloud documents. Drop one of them.")
            if declared == "cloud" and cloud_key is None:
                raise PageIndexAPIError(
                    f"{pinned} pins cloud documents — that index= "
                    "configures the local store. Drop it, or use "
                    "PageIndexLocalClient."
                    if pinned else
                    'mode="cloud" disagrees with index= — that index '
                    "configures the local store. Drop one of them.")
            if callable(cloud_key):
                cloud_key = cloud_key()
        else:
            cloud_key = api_key
            if declared == "local" and api_key is not None:
                raise PageIndexAPIError(
                    'mode="local" conflicts with api_key — an API key '
                    "means cloud documents. Drop one of them.")
            if declared == "cloud" and cloud_key is None:
                cloud_key = _env_cloud_key('mode="cloud"')
            index_conf = {name: value for name, value in index_flat.items()
                          if name != "api_key"}
        if chat is not None:
            chat_mode, chat_conf = _resolve_chat_slot(chat)
        else:
            chat_mode = "own" if chat_flat else None
            chat_conf = chat_flat
        # Every spelling lands here: strings are stripped, wrong types and
        # empty values refuse loudly — an empty chat-side value must never
        # silently select own-model chat.
        for side, slot, conf in (("index", index, index_conf),
                                 ("chat", chat, chat_conf)):
            for name, value in conf.items():
                # Slot keys are the flat names with the side prefix off.
                shown = (f'{side}["{name.removeprefix(side + "_")}"]'
                         if slot is not None else name)
                if not isinstance(value, _ARG_TYPES[name]):
                    raise PageIndexAPIError(
                        f"{shown} must be a {_ARG_TYPES[name][0].__name__}, "
                        f"got {type(value).__name__}.")
                if isinstance(value, str):
                    value = conf[name] = value.strip()
                if not value and not isinstance(value, bool):
                    raise PageIndexAPIError(
                        f"{shown} is empty — it configures nothing. Pass a "
                        "real value, or drop the argument.")
                if name == "optimize" and value not in ("full", "merge", "off"):
                    raise PageIndexAPIError(
                        f'{shown} must be "full", "merge" or "off", got {value!r}.')
                if (name in ("summary_max_words", "summary_concurrency")
                        and isinstance(value, int) and value < 1):
                    raise PageIndexAPIError(
                        f"{shown} must be a positive int, got {value!r}.")

        if cloud_key is not None:
            # cloud_api is gone: this package indexes and reads locally only.
            raise PageIndexAPIError(
                "Cloud mode is not available in this package — omit api_key "
                "to use the local store.")
        else:
            if chat_mode == "managed":
                if pinned:
                    exits = (f"{pinned} pins local documents: use "
                             "PageIndexCloudClient (or PageIndexClient("
                             "api_key=...))")
                else:
                    exits = ('Go cloud (api_key=... or index="cloud")'
                             + (' and drop mode="local"' if mode is not None
                                else ""))
                raise PageIndexAPIError(
                    "The managed chat needs cloud documents — it cannot "
                    f"read the local store. {exits}, or set your own chat "
                    "model instead.")
            from .utils import ConfigLoader
            overrides = {name: value for name, value in
                         {**index_conf, **chat_conf}.items()
                         if name in ("model", "index_model", "summary_model",
                                     "chat_model", "retrieve_model")
                         and value}
            loader = ConfigLoader()
            opt = loader.load(overrides or None)
            self.model = opt.model
            self.index_model = opt.index_model
            self.summary_model = opt.summary_model
            self._chat_model = opt.chat_model
            self._chat_model_stock = (not (overrides.get("chat_model")
                                           or overrides.get("retrieve_model")
                                           or overrides.get("model"))
                                      and not _yaml_names_chat(loader))
            self.chat_backend = chat_conf.get("chat_backend")
            self.storage_path = index_conf.get("storage_path") or ".pageindex"
            from .multimodal_api import MultimodalLocalAPI as LocalAPI
            self._api = LocalAPI(
                storage_path=self.storage_path,
                model=self.model,
                summary_model=self.summary_model,
                index_backend=index_conf.get("index_backend"),
                summary_max_words=index_conf.get("summary_max_words"),
                summary_concurrency=index_conf.get("summary_concurrency"),
                use_embedded_toc=index_conf.get("use_embedded_toc", True),
                optimize=index_conf.get("optimize", "full"),
                summary_reasoning_effort=index_conf.get("summary_reasoning_effort"),
            )
            # LiteLLM's multi-second import would otherwise land on the
            # first chat call; failures resurface there with real context.
            _preload_litellm()


    @property
    def chat_model(self):
        """Your own chat model; None on a managed-chat client."""
        return self._chat_model

    @chat_model.setter
    def chat_model(self, value):
        # Any assignment is a choice; only the untouched stock default
        # is not one.
        self._chat_model = value
        self._chat_model_stock = False

    @property
    def retrieve_model(self):
        """Legacy name for ``chat_model``."""
        return self.chat_model

    @retrieve_model.setter
    def retrieve_model(self, value):
        # 0.2.9 allowed assignment; keep the write path working too.
        self.chat_model = value

    # ---------- DOCUMENT SUBMISSION ----------

    def submit_document(
        self,
        file_path: str,
        mode: Optional[str] = None,
        beta_headers: Optional[list[str]] = None,
        folder_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        """
        Submit a PDF document for processing. Returns {'doc_id': ..., 'name': ...}.

        Cloud: uploads the file; processing is asynchronous. Pass
        ``wait=True`` to block until the document is ready, or poll
        ``get_document(doc_id)['status']`` yourself.

        Local: indexes the document in this call and stores it under
        ``storage_path``. Defaults to Flash indexing: layout-based extraction,
        refined for retrieval (a deterministic merge, then an LLM expansion
        pass); node summaries, the expansion pass, and the document
        description use ``summary_model``. Pass ``mode="standard"`` for a
        full LLM-built tree (slower). ``beta_headers`` and ``folder_id`` are
        cloud-only.

        Args:
            file_path (str): Path to the PDF file.
            mode (str, optional): Processing mode. Local defaults to "flash";
                pass "standard" for a full LLM-built tree. Cloud modes are
                passed through (e.g. "mcp").
            beta_headers (list[str], optional): Cloud-only beta feature headers.
            folder_id (str, optional): Cloud-only folder (workspace) ID.
            metadata (dict, optional): Your own JSON-serializable tags for the
                document; returned in get_document/get_tree/get_ocr responses
                and list_documents entries (both modes).
            wait (bool): Return only once the document is ready for use.
                Cloud: polls status until "completed" (raises on "failed" or
                after 30 minutes). Local: indexing is synchronous already, so
                this changes nothing. Leave False to submit many documents
                concurrently and poll afterwards.

        Returns:
            dict: {'doc_id': ..., 'name': ...}. 'name' is the stored document
                name: a taken name gains a numeric suffix (name_1..name_99)
                and a UserWarning is emitted. Older cloud servers omit 'name'.
        """
        result = self._api.submit_document(
            file_path=file_path, mode=mode,
            beta_headers=beta_headers, folder_id=folder_id, metadata=metadata,
        )
        stored = result.get("name")
        if stored and stored != os.path.basename(file_path):
            warnings.warn(
                f'Document "{os.path.basename(file_path)}" was stored as '
                f'"{stored}".',
                stacklevel=2,
            )
        if wait:
            self._wait_until_ready(result["doc_id"])
        return result

    def _wait_until_ready(self, doc_id: str, timeout: float = 1800.0) -> None:
        import requests
        interval = 2.0
        deadline = time.monotonic() + timeout
        poll_failures = 0
        while True:
            try:
                status = self.get_document(doc_id).get("status")
                poll_failures = 0
            except (PageIndexAPIError, requests.RequestException) as exc:
                if getattr(exc, "status_code", None) in (401, 403, 404):
                    raise  # a definite answer, not a poll failure
                # Tolerate transient poll failures; a 30-minute wait should
                # not die on one 502 or dropped connection.
                poll_failures += 1
                if poll_failures >= 3:
                    raise PageIndexAPIError(
                        f"Could not poll document status (doc_id: {doc_id}): "
                        f"{exc}. Processing continues in the cloud — poll "
                        "get_document(doc_id) for status."
                    ) from exc
                status = None
            if status == "completed":
                return
            if status == "failed":
                raise PageIndexAPIError(
                    f"Document processing failed (doc_id: {doc_id})."
                )
            if time.monotonic() >= deadline:
                raise PageIndexAPIError(
                    f"Timed out after {int(timeout)}s waiting for document "
                    f"processing (doc_id: {doc_id}, last status: {status}). "
                    "Processing continues in the cloud — poll "
                    "get_document(doc_id) for status."
                )
            time.sleep(interval)
            interval = min(interval * 1.5, 15.0)

    # ---------- OCR FUNCTIONALITY ----------

    def get_ocr(self, doc_id: str, format: str = "page") -> dict[str, Any]:
        """
        Get OCR status and results.

        Args:
            doc_id (str): Document ID.
            format (str): 'page' for page-based results, 'node' for node-based
                results, or 'raw' for concatenated markdown.

        Returns:
            dict: {'doc_id', 'status', 'retrieval_ready', 'result', ...}.
            With 'page', result entries are {'page_index', 'markdown', ...}.

        Local: the "OCR" result is the text extracted from the PDF while
        indexing (no OCR model runs locally, so scanned/image-only PDFs have
        no local text).
        """
        return self._api.get_ocr(doc_id=doc_id, format=format)

    def get_page_content(self, doc_id: str, pages: str) -> list[dict[str, Any]]:
        """
        Get text content of specific pages.

        Args:
            doc_id (str): Document ID.
            pages (str): Page specifier — '5-7', '3,8', or '12'.

        Returns:
            list: Matching entries from get_ocr (format='page').
        """
        wanted = set(_parse_pages(pages))
        result = self.get_ocr(doc_id, format="page")
        all_pages = result["result"]
        if all_pages is None:
            raise PageIndexAPIError(
                f"Document '{doc_id}' is not ready "
                f"(status: {result.get('status', 'unknown')})"
            )
        return [p for p in all_pages if p["page_index"] in wanted]

    def get_block(self, doc_id: str, block_id: str) -> dict[str, Any]:
        """
        One layout block of a cloud document — the page, bounding box, type
        and content behind a ``block_id`` from page content or a
        block-level citation. Cloud-only: local page content has no
        blocks, so local mode raises PageIndexAPIError.

        Args:
            doc_id (str): Document ID.
            block_id (str): Block ID as page content and citations carry
                it, e.g. ``"p3_text_5"``.

        Returns:
            dict: The block as the API returns it: {'doc_id', 'page',
            'block_id', 'bbox', 'block_type', ...}. ``bbox`` is
            ``[x0, y0, x1, y1]`` in thousandths of the page's width and
            height (0-1000), origin top-left. PageIndexAPIError with
            ``status_code == 404`` when the document or the block does not
            exist.
        """
        return self._require_cloud(
            "get_block is cloud-only — local page content has no layout "
            "blocks. Create the client with an api_key to look up blocks."
        ).get_block(doc_id=doc_id, block_id=block_id)

    def get_page_image(self, doc_id: str, page: int) -> str:
        """
        A short-lived URL to one page, rendered as a JPEG. Cloud-only:
        local mode renders no page images.

        Args:
            doc_id (str): Document ID.
            page (int): 1-based page number.

        Returns:
            str: The URL. Fetch the bytes with ``requests.get(url).content``,
            or pass it to a vision model that takes image URLs.
        """
        return self._require_cloud(
            "get_page_image is cloud-only — local mode has no page-image "
            "rendering. Create the client with an api_key to get page images."
        ).get_page_image(doc_id=doc_id, page=page)

    def get_document_image(self, doc_id: str, img_id: str) -> str:
        """
        A short-lived URL to an image OCR extracted from the document.
        Cloud-only: local mode stores no images.

        Args:
            doc_id (str): Document ID.
            img_id (str): Image ID as page content carries it,
                e.g. ``"img-7.jpeg"``.

        Returns:
            str: The URL, as ``get_page_image`` returns one.
        """
        return self._require_cloud(
            "get_document_image is cloud-only — local mode has no embedded "
            "image storage. Create the client with an api_key."
        ).get_document_image(doc_id=doc_id, img_id=img_id)

    # ---------- TREE GENERATION ----------

    def get_tree(self, doc_id: str, node_summary: bool = False,
                 include_text: bool = True) -> dict[str, Any]:
        """
        Get tree generation status and results.

        Args:
            doc_id (str): Document ID.
            node_summary (bool): Include node summaries in the tree.
            include_text (bool): Include node text (default True).
                False is useful for structure-only views (saves tokens).

        Returns:
            dict: {'doc_id', 'status', 'retrieval_ready', 'result', ...} where
            result nodes are {'title', 'node_id', 'start_index', 'end_index',
            ('summary',) ('text',) 'nodes'} in both modes. Each node's summary
            describes its own pages start_index..end_index; its text is the
            part no child holds, though a parent's may share the page its
            first child starts on.
        """
        tree = self._api.get_tree(doc_id=doc_id, node_summary=node_summary,
                                  include_text=include_text)
        if tree.get("result"):
            from .utils import _subtree, remove_fields, unify_tree
            page_count = None
            if any("end_index" not in node for node in _subtree(tree["result"])):
                page_count = self._api.get_document(doc_id).get("pageNum")
            tree["result"] = unify_tree(tree["result"], page_count)
            if not include_text:
                tree["result"] = remove_fields(tree["result"], fields=["text"])
        return tree

    def get_document_structure(self, doc_id: str) -> list[dict[str, Any]]:
        """
        Get the document's tree structure without text — summaries included.

        Returns:
            list: Tree nodes with titles, page ranges, and summaries.
        """
        return self.get_tree(doc_id, node_summary=True, include_text=False)["result"]

    def is_retrieval_ready(self, doc_id: str) -> bool:
        """
        Check if a document is ready for retrieval. API errors (including a
        missing document) are reported as False; transport errors (connection
        failures, timeouts) propagate.
        """
        try:
            result = self._api.get_tree(doc_id)
            return result.get("retrieval_ready", False)
        except PageIndexAPIError:
            return False

    # ---------- RETRIEVAL (cloud-only, deprecated) ----------

    def submit_query(self, doc_id: str, query: str, thinking: bool = False) -> dict[str, Any]:
        """
        Submit a retrieval query for a document. Returns {'retrieval_id': ...}.

        Cloud-only: the cloud API marks this endpoint deprecated in favor of
        chat completions, so local mode does not implement it — raises
        PageIndexAPIError. Use ``chat()`` instead.
        """
        return self._require_cloud(
            "submit_query is cloud-only — the retrieval API is deprecated in "
            "favor of chat completions; use chat() instead."
        ).submit_query(doc_id=doc_id, query=query, thinking=thinking)

    def get_retrieval(self, retrieval_id: str) -> dict[str, Any]:
        """
        Get retrieval status and results for a submitted query.

        Cloud-only: the cloud API marks this endpoint deprecated in favor of
        chat completions, so local mode does not implement it — raises
        PageIndexAPIError. Use ``chat()`` instead.
        """
        return self._require_cloud(
            "get_retrieval is cloud-only — the retrieval API is deprecated in "
            "favor of chat completions; use chat() instead."
        ).get_retrieval(retrieval_id=retrieval_id)


    # ---------- DOCUMENT MANAGEMENT ----------

    def get_document(self, doc_id: str) -> dict[str, Any]:
        """
        Get document metadata: {'id', 'name', 'description', 'status',
        'createdAt', 'pageNum', 'folderId', 'metadata'}. Status is one of
        "queued", "processing", "completed", "failed" (local documents are
        always "completed"; local 'folderId' is always None). 'metadata'
        is your own tags from ``submit_document``, or None.

        'createdAt' is UTC with no timezone marker, in both modes. To show
        it in the user's timezone::

            from datetime import datetime, timezone
            datetime.fromisoformat(doc["createdAt"]).replace(
                tzinfo=timezone.utc).astimezone()
        """
        return self._api.get_document(doc_id=doc_id)

    def get_document_id(self, name: str) -> str:
        """
        Look up a document's ID by its name or path. A path like
        ``"Research/Papers/attention.pdf"`` is accepted: the folder
        part is stripped because document names are unique across
        the library.

        Raises PageIndexAPIError if no document with that name exists.
        """
        name = name.rsplit("/", 1)[-1] if "/" in name else name
        result = self._api.list_documents(limit=1, name=name)
        docs = result.get("documents", [])
        if docs:
            return docs[0]["id"]
        raise PageIndexAPIError(f"No document named {name!r} found.")

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        """
        Delete a PageIndex document and all its associated data.

        Returns:
            dict: {'message': 'Document deleted successfully.'}, or an empty
            dict if the cloud API responds with no body.
        """
        return self._api.delete_document(doc_id=doc_id)

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        folder_id: Optional[str] = None,
        recursive: bool = False,
    ) -> dict[str, Any]:
        """
        List documents with pagination, newest first.

        Args:
            limit (int): Maximum documents to return (1-10000).
            offset (int): Number of documents to skip.
            folder_id (str, optional): Cloud-only folder filter.
            recursive (bool): Include documents in ``folder_id``'s
                descendant folders, flattened into one list. Cloud-only;
                local libraries have no folders.

        Returns:
            dict: {'documents': [...], 'total', 'limit', 'offset'}.

        Each document carries ``path``, its folder chain rendered
        ``"Parent/Child"`` — what a flat listing otherwise loses. None
        at the library root, for a folder reached only through a share
        of something below it, and for every local document.
        """
        return self._api.list_documents(limit=limit, offset=offset,
                                        folder_id=folder_id, recursive=recursive)

    # ---------- AGENT INTEGRATION ----------

    def agent_tools(
        self, include_management: bool = False,
    ) -> list[Callable[..., str]]:
        """
        Plain functions for any agent framework (LangChain, PydanticAI, ...).
        For the OpenAI / Claude Agent SDKs, prefer ``as_openai_tools()`` /
        ``as_claude_mcp()``.

        Cloud: the full live read tool set, discovered from the PageIndex
        MCP server when this method is called — one function per tool,
        signature and docstring synthesized from the server's schemas, calls
        executed from your process over MCP. Raises PageIndexAPIError if the
        server cannot be reached. Local: the built-in tools over the local
        store (``browse_documents``, ``get_document``,
        ``get_document_structure``, ``get_page_content``).

        Each function takes JSON-serializable arguments, returns a JSON
        string (binary results such as ``get_document_image`` as a size
        stub — a string cannot carry an image; the framework adapters
        can), and reports failures inside that JSON instead of raising —
        except a cloud 401/403, a 429/5xx that outlived the bridge's
        retries, an unreachable server or a RATE_LIMITED /
        USAGE_LIMIT_REACHED tool error, which raise PageIndexAPIError.

        Args:
            include_management (bool): Also expose tools that modify the
                library. Local: adds ``remove_document``. Cloud: the URL
                is the gate — the default serves what the read-only
                endpoint (``?tools=read``) registers; True connects to
                the full ``/mcp`` list (upload, delete, ...).
        """
        from .agent_tools import build_agent_tools
        return build_agent_tools(self, include_management)


    def agent_instructions(self, *, include_management: bool = False) -> str:
        """
        Orchestration guidance for document QA agents — pass as the agent's
        system prompt (or append to your own).

        Cloud: the live instructions the PageIndex MCP server serves for
        your key's tool set, fetched over the same session as
        ``agent_tools()`` — server-side guidance updates arrive without an
        SDK release. Raises PageIndexAPIError if the server cannot be
        reached. Local: the built-in guidance for the in-process tools.
        The client's ``instructions``, if set, follow the guidance.

        Static by design: document targeting is conversation content, not
        guidance — see ``document_context()``.

        ``include_management``: fetch the guidance for the full tool set,
        matching tools built with ``include_management=True`` (cloud;
        local guidance is a single set).
        """
        from .agent_tools import _base_instructions
        return _base_instructions(self, include_management)

    def document_context(self, doc_id: Union[str, list[str]]) -> str:
        """
        Document targeting text for the first user message: the target
        documents' names and metadata, and the directive to work within
        them. ``chat(doc_id=...)`` places it for you; on the framework
        routes you own the conversation, so lead with it yourself::

            Runner.run_sync(agent, [
                {"role": "user", "content": client.document_context(doc_id)},
                {"role": "user", "content": question},
            ])

        (or prepend it to the prompt text where the framework takes a
        string). Conversation content, not system prompt: it varies per
        request, so keeping it out of the system prompt leaves the cached
        prefix stable.

        ``doc_id``: a document ID or list of IDs, as in ``chat``. Raises
        PageIndexAPIError if a document does not exist.
        """
        from .agent_tools import doc_targeting_block
        if doc_id is None:
            raise PageIndexAPIError("doc_id must be a string or a list of "
                                    "strings.")
        return cast(str, doc_targeting_block(self, doc_id))

    def citation_prompt(self, format: str = "cite") -> str:
        """
        The citation discipline for cited answers — grounding rules plus
        how each citation is written — as served by the PageIndex MCP
        server's ``cited_answer`` prompt — what own-model
        ``chat(citations=True)`` adds. Fetch it here to append to
        ``agent_instructions()`` for an agent you build with a framework,
        or to pass another format through own-model ``chat``'s
        ``instructions=`` in place of ``citations=True`` — it is
        guidance, so it belongs in the system prompt.

        ``format`` picks how a citation is written: ``"cite"`` (the
        ``<cite doc= page= block=/>`` tags PageIndex chat writes and
        renders — the default) or ``"markdown"`` (a bracketed
        ``[doc, p. N]`` reference, for hosts that strip tags); any
        other value raises. Local documents: the SDK's
        frozen copy of the same prompt (page-level — local page
        content has no blocks).
        """
        from .agent_tools import fetch_citation_prompt
        return fetch_citation_prompt(self, format or "cite")

    def get_citations(
        self,
        answer: str,
        doc_id: Optional[Union[str, list[str]]] = None,
    ) -> list[dict[str, Any]]:
        """
        The citations in a cited answer, each with the id of the document
        it names and, for a block-level citation on a cloud document, the
        block behind it from ``get_block()`` — page, bounding box, type and
        content: the managed chat's ``citations`` entries, plus ``doc_id``
        and the block's ``text``. Reads both tag formats PageIndex chat
        writes: ``<cite doc= page= block=/>`` (own-model
        ``chat(citations=True)``) and ``<doc=…;page=…;block=…>`` (the
        managed chat). The markdown format of
        ``citation_prompt()`` is prose for readers and is not parsed.

        Args:
            answer (str): The answer text, tags included.
            doc_id (str | list[str], optional): The documents the answer
                was about — what ``chat(doc_id=...)`` took. Citations name
                documents, and the ids come from here; without it your own
                library is listed. Two documents sharing a cited name
                raise PageIndexAPIError naming both ids: pass ``doc_id``
                to pick.

        Returns:
            list: One dict per distinct citation: ``{'document', 'doc_id',
            'page'}`` for a page-level citation, plus ``'block_id'`` and
            the block's fields as ``get_block()`` returns them (``'bbox'``,
            ``'block_type'``, ...) for a block-level one. A document not in
            the library keeps ``doc_id: None``; a block the document does
            not have (a model's slip), or that cannot be looked up (local
            mode, a document you cannot read), keeps its citation without
            a bbox.
        """
        if not isinstance(answer, str):
            raise PageIndexAPIError("answer must be a str — the answer text "
                                    "with its citation tags.")
        if doc_id is not None and not isinstance(doc_id, (str, list)):
            raise PageIndexAPIError("doc_id must be a string or a list of "
                                    "strings.")
        if doc_id is not None and not doc_id:
            raise PageIndexAPIError("doc_id is empty. Pass the answer's "
                                    "document ids, or omit doc_id to "
                                    "resolve against your library.")
        citations = _parse_citations(answer)
        if not citations:
            return []
        names: dict[str, list[str]] = {}
        if doc_id is not None:
            doc_ids = [doc_id] if isinstance(doc_id, str) else doc_id
            for one_id in dict.fromkeys(doc_ids):
                name = self.get_document(one_id)["name"]
                names.setdefault(name, []).append(one_id)
        else:
            from .agent_tools import _all_documents
            for doc in _all_documents(self):
                if doc.get("name") and doc.get("id"):
                    names.setdefault(doc["name"], []).append(doc["id"])
        resolved: list[dict[str, Any]] = []
        for citation in citations:
            ids = list(dict.fromkeys(names.get(citation["document"], [])))
            if len(ids) > 1:
                raise PageIndexAPIError(
                    f"{citation['document']!r} names {len(ids)} documents "
                    f"({', '.join(ids)}) — pass doc_id= to pick one.")
            entry: dict[str, Any] = {"document": citation["document"],
                                     "doc_id": ids[0] if ids else None,
                                     "page": citation["page"]}
            block_id = citation.get("block_id")
            if block_id:
                entry["block_id"] = block_id
                if entry["doc_id"]:
                    try:
                        entry.update(self.get_block(entry["doc_id"], block_id))
                    except PageIndexAPIError as exc:
                        # Local raises carry no status; 429/5xx propagate.
                        if exc.status_code not in (None, 403, 404):
                            raise
            resolved.append(entry)
        return resolved

    def resolve_citations(
        self,
        answer: str,
        doc_id: Optional[Union[str, list[str]]] = None,
    ) -> dict[str, Any]:
        """
        Display-ready citations: the answer text with citation tags
        replaced by numbered markdown links, and each citation's full
        data from ``get_citations()`` plus an anchor and index.

        Tags (``<cite doc= page= block=/>`` and ``<doc=…;page=…>``)
        become ``[[1]](#pageindex-citation-01)``, one number per distinct
        citation, so a repeated citation reuses its number. The host
        renders the anchor targets from the ``anchor`` field.

        Args:
            answer (str): The answer text, tags included.
            doc_id (str | list[str], optional): As in ``get_citations()``.

        Returns:
            dict: ``{'answer': str, 'citations': list}`` where each
            citation carries ``'anchor'``, ``'index'`` and the fields
            ``get_citations()`` returns (``'document'``, ``'doc_id'``,
            ``'page'``, and for block-level citations ``'block_id'`` plus,
            when the block could be read, ``'bbox'``, ``'block_type'``,
            ``'text'``).
        """
        entries = self.get_citations(answer, doc_id=doc_id)
        index: dict[Any, int] = {
            (c["document"], c["page"], c.get("block_id")): i
            for i, c in enumerate(_parse_citations(answer), 1)}

        def link(m: re.Match) -> str:
            i = index.get(_citation_key(m))
            if not i:
                return m.group(0)
            return (f"[[{i}]](#pageindex-citation-{i:02d})"
                    f"{m.groupdict().get('inner') or ''}")

        return {
            "answer": _CITE_TAG_RE.sub(link, _OLD_CITATION_RE.sub(link, answer)),
            "citations": [{"anchor": f"pageindex-citation-{i:02d}", "index": i,
                           **entry} for i, entry in enumerate(entries, 1)],
        }

    def folder_context(self, folder_id: str) -> str:
        """
        Folder targeting text for the first user message, placed as
        ``document_context`` is (and ahead of it, the managed chat's
        order): the folder's name and metadata, and the directive to
        discover its documents there, rendered as the managed chat renders
        its own ``folder_id``. ``chat(folder_id=...)`` places it for you.
        Cloud-only: local libraries have no folders. ``"root"`` is the
        library itself: ``""``, nothing to place, as the managed chat
        places nothing for it. Raises PageIndexAPIError if the folder does
        not exist.
        """
        from .agent_tools import folder_targeting_block
        if folder_id is None:
            raise PageIndexAPIError("folder_id must be a string.")
        return folder_targeting_block(self, folder_id) or ""

    # ---------- FOLDER MANAGEMENT ----------

    def create_folder(
        self,
        name: str,
        description: Optional[str] = None,
        parent_folder_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create a folder (workspace). Cloud-only: local mode raises
        PageIndexAPIError.
        """
        return self._require_cloud(
            "create_folder is cloud-only — folders are not supported in local "
            "mode. Create the client with an api_key to use folders."
        ).create_folder(
            name=name, description=description, parent_folder_id=parent_folder_id,
        )

    def list_folders(self, parent_folder_id: Optional[str] = None) -> dict[str, Any]:
        """
        List folders. Cloud-only: local mode raises PageIndexAPIError.
        """
        return self._require_cloud(
            "list_folders is cloud-only — folders are not supported in local "
            "mode. Create the client with an api_key to use folders."
        ).list_folders(
            parent_folder_id=parent_folder_id,
        )

    # ---------- PATH HELPERS ----------

    def _folder_paths(self) -> dict[str, str]:
        folders = {f["id"]: f for f in self.list_folders()["folders"]}
        paths = {}
        for folder_id in folders:
            names, current = [], folder_id
            while current in folders and len(names) < len(folders):
                names.append(folders[current]["name"])
                current = folders[current].get("parent_folder_id")
            paths[folder_id] = "/".join(reversed(names))
        return paths

    def get_document_path(self, doc_id: str) -> str:
        """
        A document's path: its folder's path and its name, e.g.
        ``"Research/Papers/attention.pdf"``. Just the name when the
        document sits outside the API's folders, as at the library root
        and for every local document.
        """
        doc = self.get_document(doc_id)
        folder = doc.get("folderId") and self._folder_paths().get(doc["folderId"])
        return f"{folder}/{doc['name']}" if folder else doc["name"]

    def get_folder_path(self, folder_id: str) -> str:
        """
        A cloud folder's path: its ancestors' names and its own, root
        first, e.g. ``"Research/Papers"``. Cloud-only. Raises
        PageIndexAPIError if the folder does not exist.
        """
        self._require_cloud(
            "get_folder_path is cloud-only — folders are not supported in "
            "local mode. Create the client with an api_key.")
        path = self._folder_paths().get(folder_id)
        if path is None:
            raise PageIndexAPIError(f"Folder {folder_id!r} not found.")
        return path

    def get_folder_id(self, path: str) -> str:
        """
        The ID of the cloud folder at ``path``, written as
        ``get_folder_path`` writes it, e.g. ``"Research/Papers"``.
        Cloud-only. Raises PageIndexAPIError if no folder, or more than
        one, has that path.
        """
        self._require_cloud(
            "get_folder_id is cloud-only — folders are not supported in "
            "local mode. Create the client with an api_key.")
        paths = self._folder_paths()
        ids = [fid for fid, p in paths.items() if p == path]
        if not ids and isinstance(path, str):
            ids = [fid for fid, p in paths.items() if p == path.strip("/")]
        if len(ids) != 1:
            raise PageIndexAPIError(
                f"{path!r} names {len(ids)} folders ({', '.join(ids)})."
                if ids else f"No folder at path {path!r}.")
        return ids[0]

    def _require_cloud(self, message: str):
        # cloud_api is gone: every client is local, so cloud-only calls refuse.
        raise PageIndexAPIError(message)


class PageIndexLocalClient(PageIndexClient):
    """Local mode — no api_key parameter, no cloud access."""

    _pin = "local"

    def __init__(
        self,
        *,
        index: Optional[Union[Mapping[str, Any], str]] = None,
        chat: Optional[Union[Mapping[str, Any], str]] = None,
        index_model: Optional[str] = None,
        chat_model: Optional[str] = None,
        model: Optional[str] = None,
        summary_model: Optional[str] = None,
        summary_max_words: Optional[int] = None,
        summary_concurrency: Optional[int] = None,
        summary_reasoning_effort: Optional[str] = None,
        use_embedded_toc: Optional[bool] = None,
        optimize: Optional[str] = None,
        retrieve_model: Optional[str] = None,
        storage_path: Optional[Union[str, os.PathLike[str]]] = None,
        index_backend: Optional[dict[str, Any]] = None,
        chat_backend: Optional[dict[str, Any]] = None,
        instructions: Optional[str] = None,
    ):
        super().__init__(None, index=index, chat=chat,
                         index_model=index_model, chat_model=chat_model,
                         model=model, summary_model=summary_model,
                         summary_max_words=summary_max_words,
                         summary_concurrency=summary_concurrency,
                         summary_reasoning_effort=summary_reasoning_effort,
                         use_embedded_toc=use_embedded_toc, optimize=optimize,
                         retrieve_model=retrieve_model, storage_path=storage_path,
                         index_backend=index_backend, chat_backend=chat_backend,
                         instructions=instructions)
