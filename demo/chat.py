import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave

import numpy as np
import sounddevice as sd

import asr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SPEECH = os.path.join(OUT, "speech")

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

# pause after Mara's last chunk before the mic listens again, so her room echo isn't heard as a turn
GATE_TAIL = 0.4

FILLERS = {"um", "uh", "hmm", "mm", "mhm", "erm", "ah", "oh", "huh"}

# a sentence ends where the next word starts with a capital, a label or a quote, so "just... awful" stays whole
END = re.compile(r"[.!?]+(?=\s+[A-Z\[\"'(])|\n")
LABEL = re.compile(r"\s*\[(\w+)\]\s*")

# the direction that goes in front of the text for each label; whisper wraps the text instead
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


def parse_args():
    ap = argparse.ArgumentParser(description="Chat by voice or keyboard, answered by a local LLM and spoken by Drama 3")
    ap.add_argument("--say", default=os.path.join(HERE, "build", "say.exe"))
    ap.add_argument("--backbone", default=r"D:\models\drama-3-own\llamacpp\drama-3-q4k-experts-q8-backbone.gguf")
    ap.add_argument("--mmproj", default=r"D:\models\drama-3-own\llamacpp\drama-3-mmproj-q8_0.gguf")
    ap.add_argument("--ref", default=r"D:\drama3\verify\refs\p283.codes",
                    help="neutral reference, used when a clip is missing from the bank")
    ap.add_argument("--bank", default=os.path.join(HERE, "bank"), help="folder of <affect>.codes clips")
    ap.add_argument("--no-bank", action="store_true", help="directions only, the recording is the sole reference for every mood")
    ap.add_argument("--llm", default=r"D:\drama3\third_party\llama.cpp\build\bin\llama-server.exe")
    ap.add_argument("--llm-model", default=r"D:\models\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-UD-Q4_K_XL.gguf")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--save-wavs", action="store_true", help="also write each sentence to out/speech")
    ap.add_argument("--no-llm-start", action="store_true", help="use a llama-server that is already running")
    ap.add_argument("--no-mic", action="store_true", help="typed input only")
    ap.add_argument("--asr-dir", default=r"D:\models\parakeet-tdt-0.6b-v3", help="parakeet-tdt-0.6b-v3 sherpa-onnx int8 files")
    ap.add_argument("--vad", default=r"D:\models\silero-vad\silero_vad.onnx")
    ap.add_argument("--mic-file", help="test: play this wav (resampled to 16 kHz) into the listener instead of the microphone")
    return ap.parse_args()


def start_say(args):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "GPU-dcb6b5a1-ac97-a0f2-4fb2-15932bf0fe68"
    env["TEMP"] = r"D:\tmp"
    cmd = [args.say, args.backbone, args.mmproj, "--ref", args.ref]
    log = open(os.path.join(OUT, "say.log"), "w")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env)


def start_llm(args):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "GPU-c8c5c6ec-3408-5e0f-0512-e1c30d78556a"
    cmd = [args.llm, "-m", args.llm_model, "-ngl", "99", "-c", "8192",
           "--port", str(args.port), "--jinja"]
    log = open(os.path.join(OUT, "llama-server.log"), "w")
    return subprocess.Popen(cmd, stdout=log, stderr=log, env=env)


def wait_health(url, proc):
    while True:
        if proc is not None and proc.poll() is not None:
            sys.exit("llama-server exited with code %d, see out/llama-server.log" % proc.returncode)
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)


def clean(text):
    text = re.sub(r"<(?!/?(?:emphasis|whisper)>)[^>]*>", " ", text).replace("<|", " ")
    return " ".join(text.split())


def split_label(sentence):
    m = LABEL.match(sentence)
    if m is None or m.group(1) in ("laughs", "sighs"):
        return "neutral", sentence
    label = m.group(1).lower()
    return (label if label in DIRECTIONS else "neutral"), sentence[m.end():]


# the recording goes first and the affect clip second: the bank clips are Drama 3's own renders and drift toward
# its default accent, so the recording holds the voice while the clip supplies the mood. --no-bank keeps only the
# recording and leaves the mood to the direction text, which is milder but a shorter prompt
def clip_for(args, label):
    path = os.path.join(args.bank, label + ".codes")
    if label != "neutral" and not args.no_bank and os.path.exists(path) and os.path.exists(path[:-6] + ".txt"):
        return args.ref + ";" + path
    return args.ref


def spoken(label, text):
    if label == "whisper":
        text = re.sub(r"</?whisper>", "", text)
        return "<whisper>%s</whisper>" % text
    return (DIRECTIONS[label] + " " + text).strip()


def stream_reply(url, messages):
    body = {
        "messages": messages,
        "stream": True,
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        for raw in r:
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


def send(say, line):
    say.stdin.write((line + "\n").encode("utf-8"))
    say.stdin.flush()


def read_line(say):
    line = say.stdout.readline()
    if not line:
        raise EOFError("say.exe closed its output, see out/say.log")
    return line.decode("utf-8").rstrip("\n").split("\t")


def save_wav(path, samples):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAY_RATE)
        w.writeframes(pcm.tobytes())


def render_loop(args, say, render_q, play_q):
    n = 0
    clip = args.ref
    while True:
        label, text, turn = render_q.get()
        if text is None:
            send(say, "reset")
            read_line(say)
            clip = None
        else:
            want = clip_for(args, label)
            if want != clip:
                send(say, "ref\t%s" % want)
                res = read_line(say)
                if res[0] == "ok":
                    clip = want
                    print("[ref] %s" % want, file=sys.stderr)
                else:
                    print("[ref refused] %s: %s" % (want, res[1] if len(res) > 1 else res), file=sys.stderr)
            n += 1
            send(say, "stream\t%s" % spoken(label, text))
            chunks = []
            while True:
                res = read_line(say)
                if res[0] != "pcm":
                    break
                raw = say.stdout.read(int(res[1]) * 4)
                samples = np.frombuffer(raw, dtype="<f4")
                chunks.append(samples)
                play_q.put((samples, turn))
            if res[0] == "ok":
                print("[render] %04d [%s] %s: %s frames, first chunk %s ms, %s ms" % (n, label, text, res[1], res[2], res[3]),
                      file=sys.stderr)
                if args.save_wavs and chunks:
                    save_wav(os.path.join(SPEECH, "%04d.wav" % n), np.concatenate(chunks))
            else:
                print("[say refused] %s: %s" % (text, res[1] if len(res) > 1 else res), file=sys.stderr)
        render_q.task_done()


def play_loop(play_q, gate, announce):
    # the stream stays open and plays silence between sentences; a blocking write only returns
    # once the chunk is queued, so chunks that arrive in time follow each other without a gap
    end = 0.0
    with sd.OutputStream(samplerate=SAY_RATE, channels=1, dtype="float32") as out:
        while True:
            try:
                item = play_q.get(timeout=0.05)
            except queue.Empty:
                if gate.is_set() and time.monotonic() > end + GATE_TAIL:
                    gate.clear()
                    if announce:
                        print("[mic open]", file=sys.stderr)
                continue
            if item is None:
                out.stop()
                return
            samples, turn = item
            if not gate.is_set():
                gate.set()
                if announce:
                    print("[mic muted]", file=sys.stderr)
            if not turn["played"]:
                turn["played"] = True
                print("[latency] first audio %.2f s after %s" % (time.perf_counter() - turn["t0"], turn["from"]), file=sys.stderr)
            out.write(samples)
            end = max(end, time.monotonic()) + len(samples) / SAY_RATE
            play_q.task_done()


def mic_frames():
    with sd.InputStream(samplerate=asr.RATE, channels=1, dtype="float32", blocksize=asr.FRAME) as stream:
        while True:
            data, _ = stream.read(asr.FRAME)
            yield data[:, 0]


# paced like a live mic, so the gate sees real timing
def file_frames(path):
    x = asr.read_wav(path)
    step = asr.FRAME / asr.RATE
    start = time.monotonic()
    for n, i in enumerate(range(0, len(x) - asr.FRAME + 1, asr.FRAME)):
        time.sleep(max(0.0, start + n * step - time.monotonic()))
        yield x[i:i + asr.FRAME]


def listen(frames, rec, vad, gate, turns, finite):
    def heard(seg):
        t0 = time.perf_counter()
        text = rec.transcribe(seg)
        words = re.findall(r"[a-z']+", text.lower())
        if not words or (len(words) == 1 and words[0] in FILLERS):
            return
        print("you: %s" % text)
        print("[asr] %.1f s of speech, %.2f s" % (len(seg) / asr.RATE, rec.last_s), file=sys.stderr)
        turns.put((text, t0, "end of speech"))

    muted = False
    try:
        for frame in frames:
            if gate.is_set():
                if not muted:
                    vad.reset()
                    muted = True
                continue
            muted = False
            for seg in vad.feed(frame):
                heard(seg)
        for seg in vad.flush():
            heard(seg)
    except sd.PortAudioError as e:
        print("no microphone (%s), typing still works" % e, file=sys.stderr)
        return
    if finite:
        turns.put(None)


def read_typed(turns, quit_on_eof):
    while True:
        try:
            line = input().strip()
        except EOFError:
            if quit_on_eof:
                turns.put(None)
            return
        if line:
            turns.put((line, time.perf_counter(), "enter"))


def main():
    args = parse_args()
    os.makedirs(SPEECH, exist_ok=True)
    url = "http://127.0.0.1:%d" % args.port

    say = llm = None
    try:
        say = start_say(args)
        if not args.no_llm_start:
            llm = start_llm(args)
        print("logs in %s" % OUT, file=sys.stderr)
        print("loading...", file=sys.stderr)
        rec = vad = None
        if not args.no_mic:
            rec = asr.Recognizer(args.asr_dir)
            vad = asr.Vad(args.vad)
            print("asr loaded in %.2f s" % rec.load_s, file=sys.stderr)
        if read_line(say)[0] != "ready":
            sys.exit("say.exe did not come up, see out/say.log")
        wait_health(url, llm)
        print("ready", file=sys.stderr)

        render_q = queue.Queue()
        play_q = queue.Queue()
        turns = queue.Queue()
        gate = threading.Event()
        threading.Thread(target=render_loop, args=(args, say, render_q, play_q), daemon=True).start()
        player = threading.Thread(target=play_loop, args=(play_q, gate, not args.no_mic), daemon=True)
        player.start()
        if not args.no_mic:
            frames = file_frames(args.mic_file) if args.mic_file else mic_frames()
            threading.Thread(target=listen, args=(frames, rec, vad, gate, turns, bool(args.mic_file)), daemon=True).start()
        threading.Thread(target=read_typed, args=(turns, not args.mic_file), daemon=True).start()

        messages = [{"role": "system", "content": SYSTEM}]
        while True:
            item = turns.get()
            if item is None:
                render_q.join()
                play_q.put(None)
                player.join()
                break
            line, t0, since = item
            if line == "/reset":
                messages = messages[:1]
                render_q.put((None, None, None))
                continue

            messages.append({"role": "user", "content": line})
            turn = {"t0": t0, "from": since, "played": False}
            buf = ""
            answer = ""

            def speak(sentence):
                label, sentence = split_label(clean(sentence))
                if any(c.isalnum() for c in sentence):
                    render_q.put((label, sentence, turn))

            try:
                for piece in stream_reply(url, messages):
                    if not answer:
                        print("[latency] first token %.2f s after %s" % (time.perf_counter() - t0, since), file=sys.stderr)
                    print(piece, end="", flush=True)
                    answer += piece
                    buf += piece
                    while (m := END.search(buf)) is not None:
                        speak(buf[:m.end()])
                        buf = buf[m.end():]
            except urllib.error.HTTPError as e:
                print("llama-server said %d: %s" % (e.code, e.read().decode("utf-8", "replace")), file=sys.stderr)
                messages.pop()
                continue
            speak(buf)
            print()
            messages.append({"role": "assistant", "content": answer})
    except KeyboardInterrupt:
        pass
    finally:
        if say is not None:
            try:
                send(say, "quit")
                say.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                say.kill()
        if llm is not None:
            llm.terminate()
            try:
                llm.wait(timeout=10)
            except subprocess.TimeoutExpired:
                llm.kill()


if __name__ == "__main__":
    main()
