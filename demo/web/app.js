const talk = document.getElementById("talk");
const scope = document.getElementById("scope");
const stateEl = document.getElementById("state");
const log = document.getElementById("log");
const form = document.getElementById("type");
const line = document.getElementById("line");
const css = getComputedStyle(document.documentElement);
const g = scope.getContext("2d");
const bins = new Uint8Array(64);

let ws, ctx, loaded, micAn, outAn, mic;
let next = 0;
let state = "waking up";

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => {
    if (typeof e.data !== "string") {
      if (new Uint8Array(e.data, 0, 1)[0] === 1) play(e.data);
      return;
    }
    const m = JSON.parse(e.data);
    if (m.type === "ready") {
      // every connection starts a fresh conversation on the server
      log.replaceChildren();
      talk.disabled = false;
      show("listening");
    } else if (m.type === "state") {
      show(m.value);
    } else if (m.type === "you" || m.type === "mara") {
      add(m.type, m.text, m.label);
    } else if (m.type === "busy") {
      ws.onclose = null;
      show(m.text);
    }
  };
  ws.onclose = () => {
    talk.disabled = true;
    stop();
    show("lost the connection, trying again");
    setTimeout(connect, 2000);
  };
}

function show(s) {
  state = s;
  stateEl.textContent = s === "listening" && !mic ? "press the button and talk" : s;
}

function audio() {
  if (!ctx) {
    ctx = new AudioContext();
    loaded = ctx.audioWorklet.addModule("worklet.js");
    micAn = ctx.createAnalyser();
    outAn = ctx.createAnalyser();
    micAn.fftSize = outAn.fftSize = 128;
    outAn.connect(ctx.destination);
  }
  ctx.resume();
}

async function start() {
  audio();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
  });
  await loaded;
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "mic");
  node.port.onmessage = (e) => ws.readyState === WebSocket.OPEN && ws.send(e.data);
  src.connect(micAn);
  src.connect(node);
  // its output is silence, but a worklet that leads nowhere is not run in every browser
  node.connect(ctx.destination);
  mic = { stream, src, node };
}

function stop() {
  if (!mic) return;
  mic.stream.getTracks().forEach((t) => t.stop());
  mic.src.disconnect();
  mic.node.disconnect();
  mic = null;
}

function play(data) {
  if (!ctx) return;
  const x = new Float32Array(data.slice(1));
  const buf = ctx.createBuffer(1, x.length, 44100);
  buf.copyToChannel(x, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(outAn);
  next = Math.max(next, ctx.currentTime + 0.05);
  src.start(next);
  next += buf.duration;
}

function add(who, text, label) {
  let p = log.lastElementChild;
  if (who === "you" || !p || p.className !== "mara") {
    p = document.createElement("p");
    p.className = who;
    log.append(p);
  } else {
    p.append(" ");
  }
  if (who === "you") {
    p.textContent = text;
  } else {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = label;
    p.append(tag);
    for (const part of text.split(/(<emphasis>.*?<\/emphasis>|\[\w+\])/)) {
      if (part.startsWith("<emphasis>")) {
        const em = document.createElement("em");
        em.textContent = part.slice(10, -11);
        p.append(em);
      } else if (part.startsWith("[")) {
        const aside = document.createElement("span");
        aside.className = "aside";
        aside.textContent = part.slice(1, -1);
        p.append(aside);
      } else if (part) {
        p.append(part.replace(/<\/?\w+>/g, ""));
      }
    }
  }
  form.scrollIntoView({ block: "end", behavior: "smooth" });
}

function draw() {
  requestAnimationFrame(draw);
  const w = Math.round(scope.clientWidth * devicePixelRatio);
  const h = Math.round(scope.clientHeight * devicePixelRatio);
  if (scope.width !== w || scope.height !== h) {
    scope.width = w;
    scope.height = h;
  }
  g.clearRect(0, 0, w, h);
  const speaking = ctx && ctx.currentTime < next;
  const an = speaking ? outAn : mic ? micAn : null;
  if (an) an.getByteFrequencyData(bins);
  g.fillStyle = css.getPropertyValue(speaking ? "--accent" : "--dim");
  // low frequencies in the middle, higher ones towards the edges
  const n = 48;
  const step = w / n;
  for (let i = 0; i < n; i++) {
    const v = an ? bins[1 + Math.floor(Math.abs(i - (n - 1) / 2))] / 255 : 0;
    const bh = Math.max(3 * devicePixelRatio, v * v * h * 0.95);
    g.beginPath();
    g.roundRect(i * step + step * 0.3, (h - bh) / 2, step * 0.4, bh, step * 0.2);
    g.fill();
  }
}

talk.onclick = async () => {
  if (mic) {
    stop();
  } else {
    try {
      await start();
    } catch (err) {
      show("no microphone (" + err.message + "), you can still type");
      return;
    }
  }
  talk.classList.toggle("on", !!mic);
  talk.textContent = mic ? "Stop" : "Talk to Mara";
  show(state);
};

form.onsubmit = (e) => {
  e.preventDefault();
  const text = line.value.trim();
  if (!text || talk.disabled) return;
  audio();
  ws.send(JSON.stringify({ type: "text", text }));
  line.value = "";
};

document.getElementById("reset").onclick = (e) => {
  e.preventDefault();
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "reset" }));
  log.replaceChildren();
};

connect();
draw();
