"""Render the reference bank: one clip per affect, cloned from the p283 recording.

Pass affect names to render only those. Run it with CUDA_VISIBLE_DEVICES set to the Drama 3 card and TEMP=D:\tmp, then listen to the clips.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(HERE, "bank")
VERIFY = r"D:\drama3\verify"
PYTHON = os.path.join(VERIFY, ".venv", "Scripts", "python.exe")

# affect: (text sent to say.exe, what is actually said)
LINES = {
    "neutral": (
        "I looked through the logs this morning and everything has been running normally since the restart, so I think we can leave it alone for now.",
        None,
    ),
    "warm": (
        "[warm] Oh, that's really lovely to hear. I'm so glad it all worked out for you, and I hope you get to enjoy it properly.",
        None,
    ),
    "amused": (
        "[amused] Well, I did not [laughs] expect that. Honestly it's the funniest thing I've heard all week, so thank you for telling me.",
        "Well, I did not expect that. Honestly it's the funniest thing I've heard all week, so thank you for telling me.",
    ),
    "excited": (
        "[excited] Wait, you actually got it? That is amazing news! I can't believe it worked, so tell me everything about how it happened.",
        None,
    ),
    "sympathetic": (
        "[sympathetic, gentle] I'm so sorry to hear that. It sounds like it has been a really hard week, and I'm here if you want to talk about it.",
        None,
    ),
    "serious": (
        "[serious, measured] Before you run that, please make a backup first. It drops the whole table, and the data cannot be recovered afterwards.",
        None,
    ),
    "hesitant": (
        "[unsure, slowing down] Um, I think it was Tuesday? Maybe Wednesday, I would have to check the calendar.",
        None,
    ),
    "apologetic": (
        "[apologetic] I'm really sorry, I got that wrong earlier, and I should have checked before I told you. Let me try it again properly.",
        None,
    ),
    "emphatic": (
        "I said the <emphasis>second</emphasis> file, not the first one, so could you please open that one instead and then try it again for me.",
        "I said the second file, not the first one, so could you please open that one instead and then try it again for me.",
    ),
    "whisper": (
        "<whisper>Don't tell anyone, but I already finished the whole report last night, and I'm going to pretend I'm still working on it all week.</whisper>",
        "Don't tell anyone, but I already finished the whole report last night, and I'm going to pretend I'm still working on it all week.",
    ),
}


def strip_direction(text):
    if text.startswith("["):
        text = text[text.index("]") + 1:]
    return text.strip()


def main():
    os.makedirs(BANK, exist_ok=True)
    cmd = [os.path.join(HERE, "build", "say.exe"),
           r"D:\models\drama-3-own\llamacpp\drama-3-q4k-experts-q8-backbone.gguf",
           r"D:\models\drama-3-own\llamacpp\drama-3-mmproj-q8_0.gguf",
           "--ref", os.path.join(VERIFY, "refs", "p283.codes"), "--seed", "42"]
    say = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    if say.stdout.readline().strip() != "ready":
        sys.exit("say.exe did not come up")
    for name, (text, spoken) in LINES.items():
        if len(sys.argv) > 1 and name not in sys.argv[1:]:
            continue
        wav = os.path.join(BANK, name + ".wav")
        say.stdin.write("reset\n")
        say.stdin.write("say\t%s\t%s\n" % (wav, text))
        say.stdin.flush()
        say.stdout.readline()
        res = say.stdout.readline().rstrip("\r\n").split("\t")
        print(name, *res)
        if res[0] != "ok":
            continue
        spoken = spoken or strip_direction(text)
        subprocess.run([PYTHON, "encode_reference.py", r"D:\models\drama-3", wav,
                        os.path.join(BANK, name + ".codes"), "--text", spoken], cwd=VERIFY, check=True)
    say.stdin.write("quit\n")
    say.stdin.flush()
    say.wait()


if __name__ == "__main__":
    main()
