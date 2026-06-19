"""AI 主播 YouTube Shorts 自動化：聊天室議題 → 講稿 → 語音 → 短影音 → 上傳。

題材來源是 hf_space/ai_chat.py「AI 圓桌會議」每天產生的議題（不是量化選股清單）：
  - 自動議題：topic_rfc_macd_ib（RFC MACD IB 持倉）、topic_radar_top3（AI 雷達 Top3）
  - 使用者自訂議題：topic_user_xxxx
每個議題都已經是「籌碼AI + 新聞AI + 主持人AI」討論完、且每個角色 prompt 都內建
「絕對不能給買賣建議」規則的結果，存在 HF Dataset：
  db/chat/{date}/index.json          → 當日議題摘要列表
  db/chat/{date}/topic_{id}.json     → 單一議題完整內容（title/summary/discussion/qa）
這支只負責讀出來、轉成短影音，不重新觸發圓桌會議。當天沒有任何議題就直接跳過，不產生影片。

先以短影音（YouTube Shorts，直式 1080x1920，每支 <=60 秒）為主，不是長片：
**一個議題 = 一支短影音**，每支內部分三小段（開場/本文/結尾），各段語音各自存檔，
影片片段時長＝該段語音的真實長度（不是憑感覺猜的秒數），音畫保證對齊。

管線：
  Stage 1 題材   : load_today_material(date_str) 讀當日所有議題
  Stage 2 講稿   : 用 Gemini 把議題的 summary/discussion 改寫成 <=60秒的短影音講稿，
                  最優先原則是只能客觀轉述會議揭露的事實，絕對不能做買賣建議
                  （會議本身已經內建合規規則，這裡是第二層保險）；
                  失敗或沒有 GEMINI_API_KEY 時退回直接讀 summary，不會讓管線掛掉
  Stage 3 語音   : edge-tts 逐段產生語音
  Stage 4 影片   : 議題文字裡如果有提到本地資料庫裡真實存在的股票代號，
                  就用 mplfinance 自動畫該股K線圖當背景；沒有就用純色背景。
                  Pillow 疊標題/字幕燒進圖，每段影片時長＝語音真實長度
  Stage 5 上傳   : YouTube Data API，OAuth 用快取的 token.json 靜默刷新，
                  預設 privacyStatus="private"，人工確認後再手動切公開

部署在 GitHub Actions（雲端每天自動跑），本機也要能單獨測試，所以：
  - 所有路徑/憑證都走環境變數或固定相對路徑，本機/雲端共用同一份程式碼
  - db/price、db/info 在雲端是空的，需要先 hf_sync.pull(["db/price", "db/info"])
    （這支本身不呼叫 hf_sync，交給 scripts/run_ai_youtuber.py 負責，本機開發已經有 db/ 就不需要）
  - 中文字型：macOS 找 PingFang.ttc，Linux（GitHub Actions）需要先
    `apt-get install fonts-noto-cjk`，見 _FONT_CANDIDATES

需要你額外準備、程式無法自己生成的東西：
  - Google Cloud OAuth Client（類型選 Desktop app）下載的 client_secret.json，
    放到 db/youtube/client_secret.json（或用環境變數 YOUTUBE_CLIENT_SECRET_PATH 指定路徑）
  - 第一次在本機執行 upload_video() 會跳出瀏覽器要你手動登入一次，產生 db/youtube/token.json，
    之後本機/雲端都用這個 token.json 靜默刷新；雲端要把 client_secret.json／token.json 的內容
    存成 GitHub Secrets（YOUTUBE_CLIENT_SECRET_JSON / YOUTUBE_TOKEN_JSON），由 workflow 寫回檔案
  - ⚠️ Google OAuth consent screen 如果還是「Testing」狀態，refresh token 7 天就會過期，
    雲端排程會每週開始失敗一次，要去 Google Cloud Console 把它改成「Production」才能長期穩定跑
  - .env 要有 GEMINI_API_KEY 才能用 Stage2 的 LLM 改寫（沒有就自動退回直接讀會議摘要）
  - pip install edge-tts moviepy google-api-python-client google-auth-oauthlib google-auth-httplib2 httpx
    （mplfinance / pillow / pandas / python-dotenv / google-genai 這個環境已經裝了）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone

import edge_tts
import httpx
import mplfinance as mpf
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build as build_youtube
from googleapiclient.http import MediaFileUpload
from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

from j1stools import parquet_db

load_dotenv()

_TW = timezone(timedelta(hours=8))

_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_gemini = genai.Client(api_key=_GEMINI_API_KEY) if _GEMINI_API_KEY else None
_SCRIPT_MODEL = "gemini-2.5-flash"

REPO_ID = os.environ.get("HF_REPO_ID", "raywu918python/j1s-data")
_HF_TOKEN = os.environ.get("HF_TOKEN", "")

DEFAULT_VOICE = "zh-TW-HsiaoChenNeural"

# YouTube Shorts：直式、<=60 秒
FRAME_SIZE = (1080, 1920)
HEADER_BAND_H = 120
CHART_SIZE = (1080, 1400)
CHART_Y = 150
CAPTION_MAX_LINES = 5
_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",  # macOS（本機測試）
    "/System/Library/Fonts/STHeiti Light.ttc",  # macOS 備選
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux / GitHub Actions
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]

_YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]  # 上傳 + 改公開狀態都要
_CLIENT_SECRET_PATH = os.environ.get("YOUTUBE_CLIENT_SECRET_PATH", "db/youtube/client_secret.json")
_TOKEN_PATH = os.environ.get("YOUTUBE_TOKEN_PATH", "db/youtube/token.json")

INTRO_TEMPLATE = "歡迎收看 AI 圓桌快報，今天來看「{title}」。"
OUTRO_TEXT = "以上內容是會議揭露的客觀資訊整理，僅供研究參考，不構成投資建議。"
DISCLAIMER = "本影片由 AI 模型自動生成，內容是「AI 圓桌會議」討論的客觀資訊整理，僅供研究參考，不構成投資建議。"

_STOCK_ID_RE = re.compile(r"\b\d{4}\b")


# ─── Stage 1：題材（HF 上的當日議題） ────────────────────────────────────────


def _hf_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{path}"


def list_today_topics(date_str: str) -> list[dict]:
    """讀 db/chat/{date}/index.json：當日議題摘要列表（不含完整 discussion）。"""
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
    try:
        r = httpx.get(_hf_url(f"db/chat/{date_str}/index.json"), headers=headers, follow_redirects=True, timeout=15)
        r.raise_for_status()
        return json.loads(r.content)
    except Exception:
        return []


def load_topic(topic_id: str, date_str: str) -> dict | None:
    """讀 db/chat/{date}/topic_{id}.json：單一議題完整內容。"""
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
    try:
        r = httpx.get(
            _hf_url(f"db/chat/{date_str}/topic_{topic_id}.json"), headers=headers, follow_redirects=True, timeout=15
        )
        r.raise_for_status()
        return json.loads(r.content)
    except Exception:
        return None


def load_today_material(date_str: str) -> list[dict]:
    """讀當日所有議題的完整內容（圓桌自動議題 + 使用者自訂議題）。沒有議題回空清單。"""
    index = list_today_topics(date_str)
    topics = []
    for entry in index:
        topic = load_topic(entry["topic_id"], date_str)
        if topic:
            topics.append(topic)
    return topics


# ─── Stage 2：講稿 ──────────────────────────────────────────────────────────


def _topic_full_text(topic: dict) -> str:
    """只取這個議題自己的發言，排除 context_loader_node 廣播的「今日圓桌主題」系統訊息。

    那則系統訊息會把當天兩個自動議題（RFC持倉/雷達Top3）的股票代號都列出來，
    廣播進每一個議題的 discussion 裡；如果不排除，這裡會抓到「別的議題」的股票代號
    （例如雷達Top3議題抓到RFC持倉提到的股票），畫出不相關的K線圖。
    """
    messages = " ".join(
        m.get("message", "") for m in topic.get("discussion", []) if m.get("type") != "system"
    )
    return f"{topic.get('summary', '')} {messages}"


def _known_stocks() -> tuple[set[str], dict[str, str]]:
    info = parquet_db.query_stock_info()
    ids = set(info["stock_id"].astype(str))
    names = dict(zip(info["name"].astype(str), info["stock_id"].astype(str)))
    return ids, names


def _extract_stock_id(topic: dict, known_ids: set[str], known_names: dict[str, str]) -> str | None:
    """議題討論裡有提到真實存在的股票就回傳代號，沒有就回 None（背景用純色）。

    像「AI 雷達 Top3」這種議題討論常常只提公司名（文曄、安碁…）不提代號，
    先抓 4 位數代號，沒抓到再用公司名比對；公司名取文字裡「最早出現」的那個，結果才穩定。
    """
    text = _topic_full_text(topic)
    for match in _STOCK_ID_RE.findall(text):
        if match in known_ids:
            return match
    hits = [(text.find(name), sid) for name, sid in known_names.items() if name and name in text]
    return min(hits)[1] if hits else None


def _template_topic_text(topic: dict) -> str:
    return topic.get("summary") or "（本議題目前沒有會議摘要）"


_CJK_RE = re.compile(r"[一-鿿]")


def _looks_like_chinese_narration(text: str) -> bool:
    """Gemini 偶爾會回「好的，這是您要的講稿…」之類的英文/中介回應，不是真正的逐字稿。

    用中文字元比例擋掉這種情況，比針對特定開場白字串做字串比對更通用、更不會漏。
    """
    if not text:
        return False
    return len(_CJK_RE.findall(text)) / len(text) >= 0.5


def _llm_topic_text(topic: dict) -> str | None:
    """請 Gemini 把議題的會議記錄改寫成 <=60秒短影音講稿；失敗回 None 讓呼叫端退回會議摘要。"""
    if _gemini is None:
        print("沒有設定 GEMINI_API_KEY，講稿改用會議摘要")
        return None

    discussion_lines = [f"[{m.get('nickname', '?')}] {m.get('message', '')}" for m in topic.get("discussion", [])]
    prompt = f"""你是 AI 主播，要把一段「AI圓桌會議」的討論內容講給觀眾聽，做成一支 60 秒以內的短影音。

最優先原則（絕對不能違反，違反就是違法）：
只能客觀轉述會議中揭露的事實與數據，絕對不能做任何買賣或操作建議，不能暗示「值得投資」
「可以進場」「建議加碼」，也不能對未來股價做預測或保證。會議本身的發言已經受過合規把關，
你的工作只是用主播口吻轉述重點，不要加上自己的判斷或推論。

其他規則：
1. 只能根據以下會議記錄改寫，不要加入你自己的知識或臆測
2. 整段輸出控制在 120 字以內（含標點），這是要做成短影音的口播稿，不是長篇報導
3. 用主播播報新聞重點的口語節奏，可以用「第一、第二」「首先、另外」這種口語條列方式
   組織重點，但不要輸出條列符號（不要打 •、-、1. 這種記號，直接用文字唸出順序）
4. 一定要用繁體中文輸出
5. 直接輸出主播會唸的逐字稿本身：不要有任何開場白、前言或確認語句
   （例如不要說「好的」「這是您要的講稿」「This is a great request」），
   第一個字就必須是逐字稿內容，也不要輸出 JSON 或 markdown

議題標題：{topic.get('title', '')}
會議結論：{topic.get('summary', '')}
會議過程：
{chr(10).join(discussion_lines)}
"""
    try:
        resp = _gemini.models.generate_content(model=_SCRIPT_MODEL, contents=prompt)
        text = resp.text.strip()
        if not _looks_like_chinese_narration(text):
            print(f"議題「{topic.get('title', '')}」Gemini 回應不像中文講稿，退回會議摘要：{text[:60]!r}")
            return None
        return text
    except Exception as e:
        print(f"議題「{topic.get('title', '')}」講稿改寫失敗，退回會議摘要：{e}")
        return None


_BODY_TEXT_MAX_CHARS = 160  # 短影音口播稿上限，避免單一議題講太久；超出時截斷加「…」


def _finalize_body_text(text: str) -> str:
    """TTS 唸稿前處理：拿掉 markdown 記號/多餘換行，並限制長度，確保影片長度落在短影音區間。

    summary 是會議結論的原文，可能帶 **粗體**、編號清單，這些符號 TTS 會直接唸出來
    （例如唸出「星星」），Gemini 改寫失敗退回 summary 時尤其需要這層清理。
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _BODY_TEXT_MAX_CHARS:
        text = text[:_BODY_TEXT_MAX_CHARS].rstrip() + "…"
    return text


def build_topic_segments(topic: dict, use_llm: bool = True) -> list[dict]:
    """單一議題 → [intro, 本文, outro] 三段，對應一支短影音。"""
    title = topic.get("title") or topic.get("topic_id", "")
    body_text = (_llm_topic_text(topic) if use_llm else None) or _template_topic_text(topic)
    body_text = _finalize_body_text(body_text)
    known_ids, known_names = _known_stocks()
    stock_id = _extract_stock_id(topic, known_ids, known_names)

    body_seg = {"key": "body", "header": title, "text": body_text}
    if stock_id:
        body_seg["stock_id"] = stock_id

    return [
        {"key": "intro", "header": title, "text": INTRO_TEMPLATE.format(title=title)},
        body_seg,
        {"key": "outro", "header": title, "text": OUTRO_TEXT},
    ]


# ─── Stage 3：語音 ──────────────────────────────────────────────────────────


async def _synthesize(text: str, voice: str, out_path: str) -> None:
    await edge_tts.Communicate(text, voice=voice).save(out_path)


async def synthesize_segments(segments: list[dict], workdir: str, voice: str = DEFAULT_VOICE) -> None:
    """逐段平行產生語音，各自存檔；之後各自量真實時長去對齊畫面。"""
    tasks = []
    for seg in segments:
        seg["audio_path"] = os.path.join(workdir, f"voice_{seg['key']}.mp3")
        tasks.append(_synthesize(seg["text"], voice, seg["audio_path"]))
    await asyncio.gather(*tasks)


# ─── Stage 4：影片 ──────────────────────────────────────────────────────────


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines, current = [], ""
    for ch in text:
        trial = current + ch
        if font.getlength(trial) > max_width and current:
            lines.append(current)
            current = ch
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _fit_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    if font.getlength(text) <= max_width:
        return text
    while text and font.getlength(text + "…") > max_width:
        text = text[:-1]
    return text + "…"


def render_stock_chart(stock_id: str, workdir: str, lookback_days: int = 60) -> Image.Image | None:
    end = datetime.now(_TW)
    st = end - timedelta(days=lookback_days * 2)  # 抓寬一點再取最後 N 筆交易日，避開假日造成天數不足
    df = parquet_db.query_price([stock_id], st.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df.empty:
        return None

    df = df.tail(lookback_days).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]]

    chart_path = os.path.join(workdir, f"_chart_{stock_id}.png")
    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        volume=True,
        figsize=(CHART_SIZE[0] / 100, CHART_SIZE[1] / 100),
        savefig=dict(fname=chart_path, dpi=100),
    )
    return Image.open(chart_path).convert("RGB").resize(CHART_SIZE)


def _base_frame(stock_id: str | None, workdir: str) -> Image.Image:
    canvas = Image.new("RGB", FRAME_SIZE, (15, 20, 40))
    chart = render_stock_chart(stock_id, workdir) if stock_id else None
    if chart is not None:
        canvas.paste(chart, (0, CHART_Y))
    return canvas


def _add_caption(img: Image.Image, header: str, caption: str) -> Image.Image:
    img = img.convert("RGBA")
    w, h = img.size

    caption_font = _load_font(28)
    lines = _wrap_text(caption, caption_font, w - 40)[:CAPTION_MAX_LINES]
    band_h = 40 * len(lines) + 30

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, 0, w, HEADER_BAND_H], fill=(10, 10, 30, 235))
    draw.rectangle([0, h - band_h, w, h], fill=(0, 0, 0, 190))
    img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)
    header_font = _load_font(34)
    draw.text((20, 14), _fit_text(header, header_font, w - 40), font=header_font, fill="white")
    y = h - band_h + 15
    for line in lines:
        draw.text((20, y), line, font=caption_font, fill="white")
        y += 40

    return img.convert("RGB")


def render_segment_image(seg: dict, workdir: str) -> str:
    base = _base_frame(seg.get("stock_id"), workdir)
    framed = _add_caption(base, seg["header"], seg["text"])
    path = os.path.join(workdir, f"frame_{seg['key']}.png")
    framed.save(path)
    return path


def build_video(segments: list[dict], workdir: str, out_path: str, fps: int = 24) -> str:
    """每段影片時長＝該段語音的真實長度，音畫保證對齊，不是憑感覺猜的秒數。"""
    clips = []
    for seg in segments:
        frame_path = render_segment_image(seg, workdir)
        audio = AudioFileClip(seg["audio_path"])
        clips.append(ImageClip(frame_path).with_duration(audio.duration).with_audio(audio))

    video = concatenate_videoclips(clips, method="compose")
    video.write_videofile(out_path, fps=fps, codec="libx264", audio_codec="aac")
    video.close()
    for clip in clips:
        clip.close()
    return out_path


# ─── Stage 5：上傳 ──────────────────────────────────────────────────────────


def _youtube_client():
    """OAuth：第一次手動互動登入一次，之後都用 token.json 靜默刷新，排程才不會卡住。"""
    creds = None
    if os.path.exists(_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(_TOKEN_PATH, _YOUTUBE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(_CLIENT_SECRET_PATH):
                raise FileNotFoundError(
                    f"找不到 {_CLIENT_SECRET_PATH}，請先去 Google Cloud Console 建立 "
                    "OAuth Client（類型選 Desktop app），下載 client_secret.json 放到這個路徑"
                )
            flow = InstalledAppFlow.from_client_secrets_file(_CLIENT_SECRET_PATH, _YOUTUBE_SCOPES)
            creds = flow.run_local_server(port=0)

        os.makedirs(os.path.dirname(_TOKEN_PATH), exist_ok=True)
        with open(_TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return build_youtube("youtube", "v3", credentials=creds)


def upload_video(
    video_path: str,
    title: str,
    description: str,
    tags: list[str] | None = None,
    privacy_status: str = "private",
) -> dict:
    youtube = _youtube_client()
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags or [],
                "categoryId": "25",  # News & Politics，比科技類更貼近選股快報
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False,
                "embeddable": True,  # 沒明確設這個，頻道預設值可能是 false，前端 iframe 會放不出來
            },
        },
        media_body=MediaFileUpload(video_path, resumable=True),
    )
    return request.execute()


def save_video_log(date_str: str, entries: list[dict]) -> None:
    """把今天上傳完的影片連結存到 HF dataset，供前端讀取顯示，不用去翻 GitHub Actions log。

    跟 hf_space/ai_chat.py 的 db/chat/{date}/index.json 同樣模式：放一個固定路徑的 index.json。
    """
    if not _HF_TOKEN:
        print("沒有設定 HF_TOKEN，跳過存影片連結 log")
        return
    if not entries:
        return

    content = json.dumps(entries, ensure_ascii=False, indent=2).encode()
    try:
        from huggingface_hub import HfApi

        HfApi(token=_HF_TOKEN).upload_file(
            path_or_fileobj=content,
            path_in_repo=f"db/youtube/{date_str}/index.json",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"ai_youtuber {date_str}",
        )
        print(f"已存影片連結 log → db/youtube/{date_str}/index.json")
    except Exception as e:
        print(f"存影片連結 log 失敗：{e}")


# ─── 主流程：一個議題 = 一支短影音 ─────────────────────────────────────────


def build_topic_short(topic: dict, workdir: str, use_llm_script: bool, voice: str) -> str:
    """每段的中間檔（frame_body.png、voice_body.mp3...）用 key 命名，不含 topic_id，
    所以每個議題要存到自己專屬的子資料夾，不然同一天處理下一個議題時會互相覆蓋。"""
    topic_workdir = os.path.join(workdir, topic["topic_id"])
    os.makedirs(topic_workdir, exist_ok=True)

    segments = build_topic_segments(topic, use_llm=use_llm_script)
    asyncio.run(synthesize_segments(segments, topic_workdir, voice=voice))
    video_path = os.path.join(workdir, f"short_{topic['topic_id']}.mp4")
    build_video(segments, topic_workdir, video_path)
    return video_path


def main(
    date_str: str | None = None,
    use_llm_script: bool = True,
    voice: str = DEFAULT_VOICE,
    privacy_status: str = "unlisted",  # 前端要 embed 給訪客看，private 影片內嵌會失敗
    upload: bool = True,
    include_user_topics: bool = False,
    max_topics: int = 5,
) -> list[dict]:
    date_str = date_str or datetime.now(_TW).strftime("%Y-%m-%d")
    topics = load_today_material(date_str)

    # 預設只做兩個自動議題（RFC持倉/雷達Top3），不把使用者隨手問的議題自動公開上片：
    # 量不固定（活躍時一天可能有好幾個）、內容也沒有先天的「適合公開播報」保證。
    if not include_user_topics:
        topics = [t for t in topics if t.get("source") == "auto"]
    topics = topics[:max_topics]

    if not topics:
        print(f"{date_str} 沒有議題資料，今天不產生影片")
        return []

    workdir = f"db/youtube/{date_str}"
    os.makedirs(workdir, exist_ok=True)

    print(f"── Stage1 題材：{date_str}　{len(topics)} 個議題 ──")
    for t in topics:
        print(f"  {t['topic_id']}：{t.get('title', '')}")

    results = []
    for topic in topics:
        print(f"── 製作短影音：{topic.get('title', topic['topic_id'])} ──")
        video_path = build_topic_short(topic, workdir, use_llm_script, voice)
        print(f"   完成：{video_path}")

        if not upload:
            results.append({"topic_id": topic["topic_id"], "video_path": video_path})
            continue

        title = f"{date_str} AI圓桌快報｜{topic.get('title', topic['topic_id'])} #shorts"[:100]
        description = DISCLAIMER + "\n\n#shorts #AI選股 #台股"
        response = upload_video(
            video_path, title, description, tags=["台股", "AI選股", "shorts"], privacy_status=privacy_status
        )
        print(f"   上傳完成：https://youtu.be/{response['id']}（{privacy_status}）")
        results.append({"topic_id": topic["topic_id"], "video_path": video_path, "youtube": response})

    if upload:
        log_entries = [
            {
                "topic_id": r["topic_id"],
                "title": next(t.get("title", "") for t in topics if t["topic_id"] == r["topic_id"]),
                "video_id": r["youtube"]["id"],
                "url": f"https://youtu.be/{r['youtube']['id']}",
                "embed_url": f"https://www.youtube.com/embed/{r['youtube']['id']}",
                "privacy_status": privacy_status,
                "uploaded_at": datetime.now(_TW).strftime("%Y-%m-%d %H:%M"),
            }
            for r in results
            if "youtube" in r
        ]
        save_video_log(date_str, log_entries)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 主播圓桌快報：議題 → 講稿 → 語音 → 短影音 → 上傳")
    parser.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD，預設今天（台北時間）")
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--privacy", default="unlisted", choices=["private", "unlisted", "public"])
    parser.add_argument("--no-llm", action="store_true", help="講稿只用會議摘要，不呼叫 Gemini")
    parser.add_argument("--no-upload", action="store_true", help="只產生影片，不上傳 YouTube")
    parser.add_argument("--include-user-topics", action="store_true", help="連使用者自訂議題也做成短影音（預設只做自動議題）")
    args = parser.parse_args()

    main(
        date_str=args.date,
        use_llm_script=not args.no_llm,
        voice=args.voice,
        privacy_status=args.privacy,
        include_user_topics=args.include_user_topics,
        upload=not args.no_upload,
    )
