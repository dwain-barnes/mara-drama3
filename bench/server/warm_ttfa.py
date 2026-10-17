import http.client
import json
import statistics
import time
import wave

HOST, PORT = '127.0.0.1', 8035
TEXTS = {
    'short': 'Yeah, that works.',
    'long': 'I had a look at the logs this morning and the server has been fine since the restart, so I think we can leave it for now.',
}
ROUNDS = 6

conn = http.client.HTTPConnection(HOST, PORT, timeout=300)
rows = []

def run(name, text, n):
    body = json.dumps({'model': 'drama-3', 'input': text, 'stream': True,
                       'response_format': 'pcm', 'stream_format': 'audio',
                       'max_tokens': 500, 'seed': 123}).encode()
    t0 = time.perf_counter()
    conn.request('POST', '/v1/audio/speech', body, {'Content-Type': 'application/json'})
    res = conn.getresponse()
    if res.status != 200:
        raise SystemExit(res.read().decode(errors='replace'))
    chunks, first = [], None
    while True:
        c = res.read1(8192)
        if not c:
            break
        if first is None:
            first = (time.perf_counter() - t0) * 1000
        chunks.append(c)
    total = (time.perf_counter() - t0) * 1000
    pcm = b''.join(chunks)
    with wave.open(f'{name}-{n}.wav', 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(pcm)
    rows.append((name, n, first, total, len(pcm) / 88200))
    print(f'{name} {n} ttfa={first:.0f}ms total={total:.0f}ms audio={len(pcm) / 88200:.2f}s', flush=True)

for n in range(1, ROUNDS + 1):
    for name, text in TEXTS.items():
        run(name, text, n)

for name in TEXTS:
    t = [r[2] for r in rows if r[0] == name]
    print(f'{name}: first={t[0]:.0f} median_all={statistics.median(t):.0f} median_after_first={statistics.median(t[1:]):.0f}')
json.dump(rows, open('client_rows.json', 'w'))
