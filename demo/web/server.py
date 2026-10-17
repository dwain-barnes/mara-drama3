import argparse
import asyncio
import json
import mimetypes
import os
import re
import subprocess
import sys
import time

import aiohttp
import numpy as np
from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
OUT = os.path.join(DEMO, "out")
sys.path.insert(0, DEMO)

import asr  # noqa: E402

# Windows can map .js to text/plain in the registry, and the browser refuses to load a worklet served that way
mimetypes.add_type("text/javascript", ".js")

SYSTEM = (
    "You are Mara, and you are talking out loud with a friend, not writing. You are quick, "
    "warm and a little mischievous: you get genuinely excited about small things, you tease "
    "gently, you laugh at your own jokes, and when something matters you drop the games and "
    "say it straight. You lean in and drop your voice to whisper a secret, an aside or "
    "something tender; you slow down and hedge when you are honestly unsure, and you own a "
    "mistake with a quick, real apology, not a speech. You do not lecture, you do not list, "
    "and you never sound like a manual. One to three short spoken sentences, the way people "
    "actually talk, contractions and all. Plain text only: no markdown, no emoji.\n"
    "Your voice follows labels. Start every sentence with exactly one, in square brackets: "
    "[neutral] [warm] [amused] [excited] [sympathetic] [serious] [hesitant] [apologetic] "
    "[emphatic] [whisper]. Let the label change as your mood moves through the answer, so a "
    "reply rarely keeps one label for all its sentences. Use [whisper] for secrets, asides "
    "and quiet tenderness, never for bad news, and [sympathetic] when someone is hurting. "
    "Put [laughs] or [sighs] inside a sentence where you would really do it, and wrap one "
    "word in <emphasis></emphasis> when you would stress it out loud.\n"
    "Example: [excited] Oh, you finally fixed it? [amused] I knew you would, [laughs] you "
    "never leave a thing alone. [whisper] Between us, I'd have given up on Tuesday.\n"
    "Your friend talks to you out loud and you get their words as text from speech recognition, "
    "so a word can be slightly off. Reply to what they meant, and do not read their words back "
    "to them."
)

SAY_RATE = 44100

# the page plays what we send, so the mic stays shut until our estimate of its playback is over, plus this
GATE_TAIL = 0.4

FILLERS = {"um", "uh", "hmm", "mm", "mhm", "erm", "ah", "oh", "huh"}

END = re.compile(r"[.!?]+(?=\s+[A-Z\[\"'(])|\n")
LABEL = re.compile(r"\s*\[(\w+)\]\s*")

DIRECTIONS = {
    "neutral": "",
    "warm": "[warm]",
    "amused": "[amused]",
    "excited": "[excited]",
    "sympathetic": "[sympathetic, gentle]",
    "serious": "[serious, measured]",
    "hesitant": "[unsure, slowing down]",
    "apologetic": "[apologetic]",
    "emphatic": "",
    "whisper": "",
}

BUSY = "Mara is talking with someone else right now. Try again in a little while."


def parse_args():
    ap = argparse.ArgumentParser(description="Talk to Mara in the browser")

    def opt(flag, env, default, **kw):
        ap.add_argument(flag, default=os.environ.get(env, default), help="env %s" % env, **kw)

    opt("--say", "SAY_BIN", os.path.join(DEMO, "build", "say.exe"))
    opt("--backbone", "BACKBONE", r"D:\models\drama-3-own\llamacpp\drama-3-q4k-experts-q8-backbone.gguf")
    opt("--mmproj", "MMPROJ", r"D:\models\drama-3-own\llamacpp\drama-3-mmproj-q8_0.gguf")
    opt("--ref", "REF", r"D:\drama3\verify\refs\p283.codes")
    opt("--bank", "BANK", os.path.join(DEMO, "bank"))
    opt("--llm", "LLM_BIN", r"D:\drama3\third_party\llama.cpp\build\bin\llama-server.exe")
    opt("--llm-model", "LLM_MODEL", r"D:\models\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-UD-Q4_K_XL.gguf")
    opt("--llm-port", "LLM_PORT", 8089, type=int)
    opt("--asr-dir", "ASR_DIR", r"D:\models\parakeet-tdt-0.6b-v3")
    opt("--vad", "VAD", r"D:\models\silero-vad\silero_vad.onnx")
    opt("--port", "PORT", 8000, type=int)
    opt("--host", "HOST", "localhost")
    opt("--n-cpu-moe", "N_CPU_MOE", 0, type=int)
    opt("--say-gpu", "SAY_GPU", "GPU-dcb6b5a1-ac97-a0f2-4fb2-15932bf0fe68")
    opt("--llm-gpu", "LLM_GPU", "GPU-c8c5c6ec-3408-5e0f-0512-e1c30d78556a")
    ap.add_argument("--no-bank", action="store_true", help="directions only, the recording is the sole reference")
    ap.add_argument("--no-llm-start", action="store_true", help="use a llama-server that is already running")
    return ap.parse_args()


def clean(text):
    text = re.sub(r"<(?!/?(?:emphasis|whisper)>)[^>]*>", " ", text).replace("<|", " ")
    return " ".join(text.split())


# a sentence without a label keeps the one before it: the model labels "No way! Seriously?" once
def split_label(sentence, last):
    m = LABEL.match(sentence)
    if m is None or m.group(1) in ("laughs", "sighs"):
        return last, sentence
    label = m.group(1).lower()
    return (label if label in DIRECTIONS else last), sentence[m.end():]


# the recording first holds the voice, the bank clip second supplies the mood (see chat.py)
def clip_for(args, label):
    path = os.path.join(args.bank, label + ".codes")
    if label != "neutral" and not args.no_bank and os.path.exists(path) and os.path.exists(path[:-6] + ".txt"):
        return args.ref + ";" + path
    return args.ref


def spoken(label, text):
    if label == "whisper":
        return "<whisper>%s</whisper>" % re.sub(r"</?whisper>", "", text)
    return (DIRECTIONS[label] + " " + text).strip()


async def send(say, line):
    say.stdin.write((line + "\n").encode("utf-8"))
    await say.stdin.drain()


async def read_line(say):
    line = await say.stdout.readline()
    if not line:
        raise EOFError("say.exe closed its output, see out/web-say.log")
    return line.decode("utf-8").rstrip("\n").split("\t")


async def push(ws, msg):
    if ws.closed:
        return
    try:
        if isinstance(msg, bytes):
            await ws.send_bytes(msg)
        else:
            await ws.send_str(json.dumps(msg))
    except ConnectionResetError:
        pass


def queue(st, q, item):
    st["jobs"] += 1
    q.put_nowait(item)


async def stream_reply(http, url, messages):
    body = {
        "messages": messages,
        "stream": True,
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with http.post(url + "/v1/chat/completions", json=body) as r:
        if r.status != 200:
            print("llama-server said %d: %s" % (r.status, await r.text()), file=sys.stderr)
            return
        async for raw in r.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            for choice in json.loads(data).get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    yield piece


async def turn_loop(st, http, url):
    messages = [{"role": "system", "content": SYSTEM}]
    while True:
        ws, line, turn = await st["turns"].get()
        try:
            if line is None:
                messages = messages[:1]
                queue(st, st["render_q"], (ws, None, None, None))
                continue
            if ws.closed:
                continue
            messages.append({"role": "user", "content": line})
            buf = answer = ""
            last = ["neutral"]

            async def speak(sentence):
                last[0], sentence = split_label(clean(sentence), last[0])
                label = last[0]
                if any(c.isalnum() for c in sentence):
                    await push(ws, {"type": "mara", "text": sentence, "label": label})
                    queue(st, st["render_q"], (ws, label, sentence, turn))

            try:
                async for piece in stream_reply(http, url, messages):
                    if not answer:
                        print("[latency] first token %.2f s after %s" % (time.perf_counter() - turn["t0"], turn["from"]),
                              file=sys.stderr)
                    answer += piece
                    buf += piece
                    while (m := END.search(buf)) is not None:
                        await speak(buf[:m.end()])
                        buf = buf[m.end():]
            except aiohttp.ClientError as e:
                print("llama-server: %s" % e, file=sys.stderr)
            if not answer:
                messages.pop()
                continue
            await speak(buf)
            messages.append({"role": "assistant", "content": answer})
        finally:
            st["jobs"] -= 1


async def render_loop(st):
    args, say = st["args"], st["say"]
    clip = args.ref
    while True:
        ws, label, text, turn = await st["render_q"].get()
        try:
            if text is None:
                await send(say, "reset")
                await read_line(say)
                clip = None
                continue
            if ws.closed:
                continue
            want = clip_for(args, label)
            if want != clip:
                await send(say, "ref\t%s" % want)
                res = await read_line(say)
                if res[0] == "ok":
                    clip = want
                else:
                    print("[ref refused] %s: %s" % (want, res[1:]), file=sys.stderr)
            await send(say, "stream\t%s" % spoken(label, text))
            while True:
                res = await read_line(say)
                if res[0] != "pcm":
                    break
                n = int(res[1])
                raw = await say.stdout.readexactly(n * 4)
                st["play_end"] = max(st["play_end"], time.monotonic()) + n / SAY_RATE
                if not turn["played"]:
                    turn["played"] = True
                    print("[latency] first audio %.2f s after %s" % (time.perf_counter() - turn["t0"], turn["from"]),
                          file=sys.stderr)
                await push(ws, b"\x01" + raw)
            if res[0] == "ok":
                print("[render] [%s] %s: %s frames, first chunk %s ms, %s ms" % (label, text, res[1], res[2], res[3]),
                      file=sys.stderr)
            else:
                print("[say refused] %s: %s" % (text, res[1:]), file=sys.stderr)
        finally:
            st["jobs"] -= 1


def gated(st):
    return time.monotonic() < st["play_end"] + GATE_TAIL


async def report_state(st, ws):
    last = None
    while not ws.closed:
        if time.monotonic() < st["play_end"]:
            state = "speaking"
        elif gated(st):
            state = "muted"
        elif st["jobs"]:
            state = "thinking"
        else:
            state = "listening"
        if state != last:
            await push(ws, {"type": "state", "value": state})
            last = state
        await asyncio.sleep(0.05)


async def hello(st, ws):
    await st["ready"].wait()
    st["vad"].reset()
    queue(st, st["turns"], (ws, None, None))
    await push(ws, {"type": "ready"})
    await report_state(st, ws)


async def heard(st, ws, seg):
    t0 = time.perf_counter()
    text = await asyncio.to_thread(st["rec"].transcribe, seg)
    words = re.findall(r"[a-z']+", text.lower())
    if not words or (len(words) == 1 and words[0] in FILLERS):
        return
    print("you: %s\n[asr] %.1f s of speech, %.2f s" % (text, len(seg) / asr.RATE, st["rec"].last_s), file=sys.stderr)
    await push(ws, {"type": "you", "text": text})
    queue(st, st["turns"], (ws, text, {"t0": t0, "from": "end of speech", "played": False}))


async def talk(request, st):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    if st["client"] is not None:
        await push(ws, {"type": "busy", "text": BUSY})
        await ws.close()
        return ws

    st["client"] = ws
    greeter = asyncio.create_task(hello(st, ws))
    pcm = np.zeros(0, dtype=np.float32)
    muted = False
    try:
        async for msg in ws:
            if not st["ready"].is_set():
                continue
            if msg.type == aiohttp.WSMsgType.BINARY:
                if gated(st):
                    if not muted:
                        st["vad"].reset()
                        pcm = pcm[:0]
                        muted = True
                    continue
                muted = False
                pcm = np.concatenate([pcm, np.frombuffer(msg.data, dtype="<i2").astype(np.float32) / 32768])
                n = len(pcm) - len(pcm) % asr.FRAME
                for i in range(0, n, asr.FRAME):
                    for seg in st["vad"].feed(pcm[i:i + asr.FRAME]):
                        await heard(st, ws, seg)
                pcm = pcm[n:]
            elif msg.type == aiohttp.WSMsgType.TEXT:
                m = json.loads(msg.data)
                text = m.get("text", "").strip()
                if m.get("type") == "text" and text:
                    await push(ws, {"type": "you", "text": text})
                    queue(st, st["turns"], (ws, text, {"t0": time.perf_counter(), "from": "typing", "played": False}))
                elif m.get("type") == "reset":
                    queue(st, st["turns"], (ws, None, None))
    finally:
        greeter.cancel()
        st["client"] = None
    return ws


async def page(request):
    return web.FileResponse(os.path.join(HERE, request.match_info["name"] or "index.html"))


async def wait_health(http, url, llm):
    while True:
        if llm is not None and llm.poll() is not None:
            sys.exit("llama-server exited with code %d, see out/web-llama-server.log" % llm.returncode)
        try:
            async with http.get(url + "/health", timeout=aiohttp.ClientTimeout(total=2)) as r:
                if r.status == 200 and (await r.json()).get("status") == "ok":
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            pass
        await asyncio.sleep(1)


async def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)

    st = dict(args=args, client=None, ready=asyncio.Event(), jobs=0, play_end=0.0,
              turns=asyncio.Queue(), render_q=asyncio.Queue())
    app = web.Application()
    app.router.add_get("/ws", lambda request: talk(request, st))
    app.router.add_get(r"/{name:(|app\.css|app\.js|worklet\.js)}", page)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    print("open http://localhost:%d" % args.port, file=sys.stderr)

    say = llm = None
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.say_gpu)
        say = await asyncio.create_subprocess_exec(
            args.say, args.backbone, args.mmproj, "--ref", args.ref, "--n-cpu-moe", str(args.n_cpu_moe),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(OUT, "web-say.log"), "w"), env=env)
        if not args.no_llm_start:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.llm_gpu)
            log = open(os.path.join(OUT, "web-llama-server.log"), "w")
            llm = subprocess.Popen([args.llm, "-m", args.llm_model, "-ngl", "99", "-c", "8192",
                                    "--port", str(args.llm_port), "--jinja"], stdout=log, stderr=log, env=env)
        print("loading, logs in %s" % OUT, file=sys.stderr)
        st["rec"] = await asyncio.to_thread(asr.Recognizer, args.asr_dir)
        st["vad"] = asr.Vad(args.vad)
        if (await read_line(say))[0] != "ready":
            sys.exit("say.exe did not come up, see out/web-say.log")
        st["say"] = say

        url = "http://127.0.0.1:%d" % args.llm_port
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as http:
            await wait_health(http, url, llm)
            st["ready"].set()
            print("ready", file=sys.stderr)
            await asyncio.gather(turn_loop(st, http, url), render_loop(st))
    finally:
        if say is not None and say.returncode is None:
            try:
                await send(say, "quit")
                await asyncio.wait_for(say.wait(), 10)
            except (OSError, asyncio.TimeoutError):
                say.kill()
        if llm is not None:
            llm.terminate()
            try:
                llm.wait(timeout=10)
            except subprocess.TimeoutExpired:
                llm.kill()
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
