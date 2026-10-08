import os
import json
import socket

# Force IPv4 resolution globally to prevent "Network is unreachable" socket errors
# in environments (like GitHub Actions runners) that lack external IPv6 routing.
_orig_getaddrinfo = socket.getaddrinfo
def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0:
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, type, proto, flags)
socket.getaddrinfo = _patched_getaddrinfo

STUDENT_ID = os.environ.get("StuId", "")
PASSWORD = os.environ.get("UISPsw", "")

WEBVPN_BASE = "https://webvpn.fudan.edu.cn"
IDP_BASE = "https://id.fudan.edu.cn"
ICOURSE_BASE = "https://icourse.fudan.edu.cn"

WEBVPN_AES_KEY = b"wrdvpnisthebest!"
WEBVPN_AES_IV = b"wrdvpnisthebest!"

TENANT_CODE = "222"
GROUP_CODE = "2095000001"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# 模型服务商配置（按列表顺序作为优先级，从前往后尝试）。
# 用户可以在这里随意添加/删除/重排服务商和模型。
# 兼容性：只设置 DASHSCOPE_API_KEY 也能跑（modelscope 项的 api_key 直接读取它）。
# 同名 provider 多次出现 → resolve_model_providers() 把它们的 models 合并到首次出
# 现的那条；这避免 Summarizer 内部按 name 索引 client 字典时被后写覆盖。
MODEL_PROVIDERS: list[dict] = [
    {
        "name": "modelscope",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "default_base_url": "https://api-inference.modelscope.cn/v1/",
        "models": [
            "deepseek-ai/DeepSeek-V4-Pro",
            "deepseek-ai/DeepSeek-V4-Pro-0813",
            "Qwen/Qwen3.5-397B-A17B",
            # 注意：ModelScope 上"有模型页但无推理 provider"的模型会恒 400
            # （如 DeepSeek-V4-Flash、MiniMax-M3、Tencent-Hunyuan/Hy3 实测空响应），
            # 加模型前必须先冒烟验证。2026-09-14。
        ],
    },
    {
        # 首选 modelscope 余额耗尽时（429 insufficient_balance）自动降级到这里。
        "name": "deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
        "models": [
            "deepseek-flash",
            "deepseek-v4-pro",
        ],
    },
    # {
    #     "name": "modelscope",
    #     "api_key_env": "DASHSCOPE_API_KEY",
    #     "base_url_env": "DASHSCOPE_BASE_URL",
    #     "default_base_url": "https://api-inference.modelscope.cn/v1/",
    #     "models": [
    #         "deepseek-ai/DeepSeek-V3.2",
    #         "ZhipuAI/GLM-5",
    #         "MiniMax/MiniMax-M2.5",
    #         "Qwen/Qwen3.5-397B-A17B",
    #     ],
    # },
    {
        "name": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": [
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
        ],
    }
]


def resolve_model_providers() -> list[dict]:
    """Resolve MODEL_PROVIDERS into runtime configs.

    Drops providers whose api_key env var is unset. Same-name entries get
    their model lists merged into the first occurrence (Summarizer's client
    dict keys on name and would otherwise collide).

    Returns:
        list of {name, api_key, base_url, models}.
    """
    resolved: list[dict] = []
    by_name: dict[str, dict] = {}
    for p in MODEL_PROVIDERS:
        api_key = os.environ.get(p["api_key_env"], "").strip()
        if not api_key:
            continue
        base_url = (
            os.environ.get(p.get("base_url_env", ""), "").strip()
            or p.get("default_base_url", "")
        )
        if not base_url:
            continue
        if p["name"] in by_name:
            existing = by_name[p["name"]]
            for m in p["models"]:
                if m not in existing["models"]:
                    existing["models"].append(m)
            continue
        entry = {
            "name": p["name"],
            "api_key": api_key,
            "base_url": base_url,
            "models": list(p["models"]),
        }
        resolved.append(entry)
        by_name[p["name"]] = entry
    return resolved


# Legacy compatibility shims (kept so other modules importing these don't break)
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# QQ SMTP (can be overridden for custom SMTP providers)
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", "")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.qq.com")
try:
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
except ValueError:
    SMTP_PORT = 465


# Database & Storage
DATA_DIR = os.environ.get("DATA_DIR", "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")  # ffmpeg-decoded f32le scratch buffers
DB_PATH = os.environ.get("DB_PATH", os.path.join(DATA_DIR, "icourse.db"))

# Sherpa-onnx ASR model directory.  Default: SenseVoice (zh+en+ja+ko+yue, int8).
# ASR_MODEL_DIR is the new name; SENSEVOICE_MODEL_DIR is the legacy env var
# kept as a fallback so existing CI cache keys keep working.
ASR_MODEL_DIR = os.environ.get(
    "ASR_MODEL_DIR",
    os.environ.get(
        "SENSEVOICE_MODEL_DIR",
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    ),
)
SENSEVOICE_MODEL_DIR = ASR_MODEL_DIR  # alias for any straggler imports
SILERO_VAD_PATH = os.environ.get("SILERO_VAD_PATH", "silero_vad.onnx")
# The CI's cached silero predates the upstream asset rename; if the file is
# missing/short (e.g. a 404 HTML got written over it once the cache misses),
# Transcriber re-fetches this URL at runtime.  Doing it in Python keeps
# check.yml untouched — a workflow-file edit would park every scheduled run
# behind action_required until someone clicks Approve in the UI.
SILERO_VAD_URL = os.environ.get(
    "SILERO_VAD_URL",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad_v4.onnx",
)
# CT-Transformer punctuation (zh/en, int8): adds ，。？！ to SenseVoice
# segments.  Bootstrapped at runtime like the VAD; a missing or unloadable
# model degrades to unpunctuated text rather than failing the lecture.
PUNCT_MODEL_DIR = os.environ.get(
    "PUNCT_MODEL_DIR",
    "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
)
PUNCT_MODEL_URL = os.environ.get(
    "PUNCT_MODEL_URL",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2",
)

# ASR backend selector — Transcriber dispatches on this.  When changing,
# ASR_MODEL_DIR must point at a matching sherpa-onnx model bundle:
#   sensevoice — sherpa-onnx-sense-voice-* (multi-lang CTC, single model)
#   firered    — sherpa-onnx-fire-red-asr2-ctc-* (CTC, single model.onnx)
#   zipformer  — sherpa-onnx-zipformer-* (transducer, encoder/decoder/joiner)
ASR_BACKEND = os.environ.get("ASR_BACKEND", "sensevoice").strip().lower()
# Inference thread count.  4 fully saturates a 4-vCPU GitHub runner.
ASR_NUM_THREADS = int(os.environ.get("ASR_NUM_THREADS", "4"))

# ── Scheduler concurrency knobs.  All overridable via env. ────────────────
# image_pool: image downloads are tiny and IO-bound, 20 saturates bandwidth
# without hammering the iCourse server.
IMAGE_WORKERS = int(os.environ.get("IMAGE_WORKERS", "20"))
# OCR pool: pool size is the hard ceiling; a fixed BoundedSemaphore(OCR_MAX_TARGET)
# gates live concurrency since RapidOCR is single-threaded CPU-bound.
OCR_MAX_WORKERS = int(os.environ.get("OCR_MAX_WORKERS", "8"))
# Fixed cap — no dynamic CPU-based adjustment.  RapidOCR is single-threaded;
# more than 2 concurrent workers don't increase throughput on 4-core runners.
OCR_MAX_TARGET = int(os.environ.get("OCR_MAX_TARGET", "2"))
# Two concurrent ffmpeg audio extractions: the current lecture being
# transcribed + one pre-decoded for the next lecture.  Bandwidth-fair sharing
# at 20 MB/s split = ~10 MB/s each.
VIDEO_DOWNLOAD_CONCURRENCY = int(
    os.environ.get("VIDEO_DOWNLOAD_CONCURRENCY", "2")
)

# 是否优先使用 iCourse 官方字幕（跳过 ASR 转录）。默认关闭。
USE_OFFICIAL_TRANSCRIPT = (
    os.environ.get("USE_OFFICIAL_TRANSCRIPT", "").strip().lower()
    in ("1", "true", "yes")
)

# 监控的课程 ID 列表
COURSE_IDS = [
    c.strip()
    for c in os.environ.get("COURSE_IDS", "").split(",")
    if c.strip()
]

# 学期级课程目录爬取（已弃用 — main.py 现在自动发现所有学期）。
# 保留此变量仅用于兼容老部署环境，新部署无需设置。
# 例：CRAWL_TERM=25
CRAWL_TERM = os.environ.get("CRAWL_TERM", "").strip()

# 强制本次运行刷新学期目录（不受每月 5/25 号限制）。
# 例：FORCE_CRAWL=true
FORCE_CRAWL = os.environ.get("FORCE_CRAWL", "").strip().lower() in ("1", "true", "yes")

# 强制全量重摘录的讲义 sub_id 列表（逗号分隔）。
# 运行前会清空这些节的 processed/error/邮件状态、缓存转录与摘要，并把其
# ppt_pages 全部重置为 pending（含 dedup_dropped），使下次处理从下载、
# ASR、PPT OCR 到摘要完整重跑。用于强制摘录被误标完成的缺失讲义。
# 例：FORCE_SUB_IDS=664140
FORCE_SUB_IDS = [
    s.strip()
    for s in os.environ.get("FORCE_SUB_IDS", "").split(",")
    if s.strip()
]

# 手动注入的音频资产。当某节录播在 iCourse 侧的视频文件坏了（如 泛函分析
# 664140：WebVPN 出口拿到的对象音轨是死的，校园网出口正常），可人工从健康
# 出口下载视频，抽出 16kHz 单声道 f32le 原始 PCM（与 CI 的 ffmpeg 输出同
# 格式，无头），传成 release 资产后在 force_audio/<sub_id>.json 登记：
#   {"sub_id": "664140", "url": "https://.../<asset>",
#    "ppt_url": "https://.../<slides.json>"}
# LectureRunner 取转录前优先使用注入音频，跳过坏视频下载。
# url 可缺省（如音轨彻底无语音、只保留课件注入的场合）。
# 新增 video_url 键：直接把浏览器/校园网出口存的**完整健康 mp4**（同样传成
# GitHub release 资产）接进来——ASR 与板书抽帧都改用它（经 get_stream_params
# 的 GitHub 直通，不绕 WebVPN），一条健康原片喂两个消费者，实现与服务端
# 出口对象的**双线验证**。
def _load_force_audio() -> dict[str, dict]:
    out: dict[str, dict] = {}
    d = os.path.join(os.getcwd(), "force_audio")
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, name)) as f:
                    entry = json.load(f)
                url = entry.get("url")
                ppt_url = entry.get("ppt_url")
                has_url = ((isinstance(url, str) and url)
                           or (isinstance(url, list) and url))
                has_ppt = isinstance(ppt_url, str) and bool(ppt_url)
                has_video = (isinstance(entry.get("video_url"), str)
                             and bool(entry.get("video_url")))
                if not has_url and not has_ppt and not has_video:
                    continue
                sid = str(entry.get("sub_id") or name.rsplit(".", 1)[0])
                out[sid] = entry
            except Exception:
                continue
    return out


FORCE_AUDIO_MAP: dict[str, dict] = _load_force_audio()

# ── 板书兜底（board-from-video）────────────────────────────────────────────
# 板书型课程（如泛函分析）的 iCourse 截图 feed 常常只有占位屏甚至空，内容
# 只在视频帧里。当某节课音轨不可用（空转写或下载截断） **且** 平台 PPT 无
# 实质文字时，与其记可重试 error 空等，不如把视频流经 ffmpeg（带 reconnect）
# 抽帧、dHash 去重、逐帧 OCR，把板书文字当课件喂给摘要（等价于 2026-09-29 对
# 664140 的手工恢复）。置空/0 关闭该兜底。
BOARD_FALLBACK = os.environ.get("BOARD_FALLBACK", "1").strip().lower() not in ("0", "false", "no", "")
# 抽帧间隔（秒）。板书内容随时间累积，20s 足够覆盖且不爆 OCR 预算。
BOARD_FRAME_INTERVAL = int(os.environ.get("BOARD_FRAME_INTERVAL", "20"))
# 单节最多送去 OCR 的板书帧数（去重后按时间等比截断）。
BOARD_MAX_PAGES = int(os.environ.get("BOARD_MAX_PAGES", "260"))

# ── 音轨验货闸（readiness gate）──────────────────────────────────────────
# 处理新课前，先对每个视频候选（含 /play/1/ 播放口路径）抽两段短窗实测音频
# 动态（probe_audio，纯 ffmpeg 流读，每候选 ~2MB、不加载 ASR）。全部平直
# = 平台混流未到位（泛函 9.21/9.28：审核解除后 WebVPN 出口音轨依然死），
# 整节挂 waiting 软状态每天重探——不吃 3 次 error 配额、不进流程、不出
# 半吊子摘要；一旦探到活音轨自动用该候选走正流程。超过 GATE_MAX_DAYS 仍
# 全哑 → 放行（板书兜底从真实画面抢救内容）。置 0 关闭。
AUDIO_GATE = os.environ.get("AUDIO_GATE", "1").strip().lower() not in ("0", "false", "no", "")
AUDIO_GATE_MAX_DAYS = int(os.environ.get("AUDIO_GATE_MAX_DAYS", "7"))
# 按课程"无限等待活音轨"名单（逗号分隔 course_id，默认 37547=泛函分析）：
# 名单内课程即使哑超过 GATE_MAX_DAYS 也**不走板书妥协**——老师口述权重高，
# 宁可一直挂着（就绪闸每日重探、前端 Waiting、零配额消耗）也不出只有板书
# 的半份摘要；学校把真混流推上 CDN 的那天，自动出完整音频+板书摘要。
HOLD_WAIT_AUDIO_COURSES = [
    s.strip()
    for s in os.environ.get("HOLD_WAIT_AUDIO_COURSES", "37547").split(",")
    if s.strip()
]
