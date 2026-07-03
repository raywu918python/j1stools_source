"""LLM 切換機制 — 獨立檔案，複製到別的專案就能直接用，不需要改任何程式碼。

外部呼叫只需要：
    from llm_provider import get_llm
    response = get_llm().invoke([...])

不用管背後是哪個帳號、哪個模型 —— 額度用完會自動 fallback 到下一個，
免費模型優先，付費/不穩定模型預設關閉，要開才會加進 fallback 鏈。

環境變數（部署環境的 Secrets 設定）
----------------------------------
免費模型（建議至少設一組 GROQ_API_KEY，否則沒有任何可用的 LLM）：
    GROQ_API_KEY              Groq 帳號 1
    GROQ_API_KEY_JUST1STOCK   Groq 帳號 2（可選，額度跟帳號1獨立，等於額度加倍）
    CEREBRES_TOKEN            Cerebras（可選，免費，Groq 全部用完才輪到它）
    HF_TOKEN                  HuggingFace Serverless（Qwen3-8B 用，見 LLM_USE_QWEN）

付費模型（LLM_USE_PAID=1 才會用到，見下方開關）：
    DEEPSEEK_API_KEY          DeepSeek（deepseek-chat，OpenAI 相容 API）
    ANTHROPIC_API_KEY         Claude（claude-haiku-4-5，走 langchain-anthropic）

開關（預設都關閉，因為兩者都不穩定/要花錢 —— 開了才會加進 fallback 鏈）：
    LLM_USE_QWEN=1   加入 HF Serverless Qwen3-8B（免費無額度上限，但
                     with_structured_output() 會壞掉，只有純文字生成/bind_tools 可以用；
                     預設關閉，避免結構化輸出的呼叫方意外壞掉）
    LLM_USE_PAID=1   加入付費模型（DeepSeek → Claude，排在免費鏈最後面，只有免費模型
                     全部用完才會付費。要加其他付費 LLM，往 _PAID_MODELS 加一筆設定即可，
                     沒設對應 API Key 的項目會自動跳過）

開發測試用（不建議在正式流量開啟 —— 會失去 fallback，額度用完直接報錯）：
    LLM_PROVIDER=openrouter   強制只用 OpenRouter 免費模型，不 fallback
    LLM_PROVIDER=qwen         強制只用 Qwen3-8B，不 fallback
"""

from __future__ import annotations

import os

from langchain_core.callbacks import BaseCallbackHandler

_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
_GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY_JUST1STOCK", "")
_CEREBRAS_TOKEN = os.environ.get("CEREBRES_TOKEN", "")
_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
_ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

_USE_QWEN_DEFAULT = os.environ.get("LLM_USE_QWEN", "0") == "1"
_USE_PAID_DEFAULT = os.environ.get("LLM_USE_PAID", "0") == "1"

# 開發測試用的手動 override，正式流量不要設（見上方 docstring）。
_manual_provider = os.environ.get("LLM_PROVIDER", "auto")

# Groq 上各自獨立速率限制的模型，依序當主力/備援（全部測試過 structured output + 工具呼叫）
_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

# Cerebras 上的模型 — 走 OpenAI 相容端點（langchain-cerebras 套件相依 langchain-core 太舊，
# 會把 langgraph/langchain-groq 需要的 1.x 版本擠掉，所以不用那個套件）。
_CEREBRAS_MODELS = ["gpt-oss-120b", "zai-glm-4.7"]

# 付費模型清單，依序當付費 fallback 的主力/備援。沒設對應 API Key 的項目
# 會在 _build_paid_llms() 自動跳過，所以複製這支檔案到別的專案不用改這裡。
# kind="openai_compatible" 走 ChatOpenAI + base_url；kind="anthropic" 走 ChatAnthropic。
_PAID_MODELS: list[dict] = [
    {"model": "deepseek-chat", "kind": "openai_compatible", "base_url": "https://api.deepseek.com", "api_key": _DEEPSEEK_API_KEY},
    {"model": "claude-haiku-4-5-20251001", "kind": "anthropic", "api_key": _ANTHROPIC_API_KEY},
]


class _FallbackLogger(BaseCallbackHandler):
    """模型呼叫失敗時印出來，方便在 Log 看到 fallback 有沒有觸發。"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        print(f"[llm_provider] 模型 {self.model_name} 失敗，切換下一個: {type(error).__name__}: {error}")


def get_provider() -> str:
    """開發測試用：目前的手動 override（auto / openrouter / qwen）。"""
    return _manual_provider


def set_provider(provider: str) -> None:
    """開發測試用：強制指定單一模型、不 fallback。設回 "auto" 恢復正常的免費自動 fallback 鏈。"""
    global _manual_provider
    if provider not in ("auto", "groq", "openrouter", "qwen"):
        raise ValueError(f"不支援的 provider: {provider}，可選 auto / groq / openrouter / qwen")
    _manual_provider = "auto" if provider == "groq" else provider


def _build_free_llms() -> list:
    """Groq（多模型 x 多帳號）+ Cerebras —— 額度都獨立，前一個用完自動換下一個。"""
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI

    keys = [k for k in (_GROQ_API_KEY, _GROQ_API_KEY_2) if k]
    combos = [(m, k) for m in _GROQ_MODELS for k in keys]
    llms = [ChatGroq(model=m, api_key=k, callbacks=[_FallbackLogger(m)]) for m, k in combos]

    if _CEREBRAS_TOKEN:
        llms += [
            ChatOpenAI(
                model=m,
                api_key=_CEREBRAS_TOKEN,
                base_url="https://api.cerebras.ai/v1",
                callbacks=[_FallbackLogger(f"cerebras:{m}")],
            )
            for m in _CEREBRAS_MODELS
        ]
    return llms


def _openrouter_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="nvidia/nemotron-nano-9b-v2:free",
        api_key=_OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


def _qwen_llm():
    """HF Serverless Inference（task="text-generation"）— 真正免費無額度上限。

    注意：不支援 with_structured_output()（會 raise NotImplementedError），
    只能用在純文字生成或 bind_tools()，呼叫方自己確認用途再開 LLM_USE_QWEN。
    """
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-8B",
        huggingfacehub_api_token=_HF_TOKEN,
        task="text-generation",
        max_new_tokens=2048,
    )
    return ChatHuggingFace(llm=endpoint)


def _build_paid_llms() -> list:
    llms = []
    for cfg in _PAID_MODELS:
        if not cfg.get("api_key"):
            continue  # 沒設對應 API Key，跳過（讓這份設定清單可以直接複製到別的專案用）
        logger = _FallbackLogger(f"paid:{cfg['model']}")
        if cfg["kind"] == "anthropic":
            from langchain_anthropic import ChatAnthropic

            llms.append(ChatAnthropic(model=cfg["model"], api_key=cfg["api_key"], callbacks=[logger]))
        else:
            from langchain_openai import ChatOpenAI

            llms.append(
                ChatOpenAI(
                    model=cfg["model"],
                    api_key=cfg["api_key"],
                    base_url=cfg.get("base_url"),
                    callbacks=[logger],
                )
            )
    return llms


def get_llm(*, use_qwen: bool | None = None, use_paid: bool | None = None, qwen_first: bool = False):
    """回傳一個包好 fallback 鏈的 LangChain LLM，外部直接 invoke() / bind_tools() 就好。

    use_qwen  : 是否把 Qwen3-8B 加進鏈（預設看 LLM_USE_QWEN 環境變數）
    use_paid  : 是否把付費模型加進鏈尾（預設看 LLM_USE_PAID 環境變數，目前 _PAID_MODELS 是空的）
    qwen_first: Qwen3-8B 排最前面（其他免費模型當它的 fallback）—— 給「流量大、
                容許偶爾結構化失敗、想省 Groq 額度」的呼叫方用（例如客服問答）。
                預設 False，即 Qwen 只當免費鏈用完後的最後備援。
    """
    if _manual_provider == "openrouter":
        return _openrouter_llm()
    if _manual_provider == "qwen":
        return _qwen_llm()

    use_qwen = _USE_QWEN_DEFAULT if use_qwen is None else use_qwen
    use_paid = _USE_PAID_DEFAULT if use_paid is None else use_paid

    llms = _build_free_llms()

    if use_qwen and qwen_first:
        llms = [_qwen_llm()] + llms
    elif use_qwen:
        llms = llms + [_qwen_llm()]

    if use_paid:
        llms += _build_paid_llms()

    if not llms:
        raise RuntimeError("沒有任何可用的 LLM，請確認環境變數至少設定一組 GROQ_API_KEY")

    primary, *fallbacks = llms
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
