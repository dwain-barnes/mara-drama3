// averages each run of input samples down to one 16 kHz sample, which is enough low-pass for speech recognition
class Mic extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / 16000;
    this.pos = 0;
    this.sum = 0;
    this.n = 0;
    this.out = new Int16Array(512);
    this.len = 0;
  }

  process([input]) {
    for (const v of input[0] || []) {
      this.sum += v;
      this.n++;
      if (++this.pos < this.step) continue;
      this.pos -= this.step;
      this.out[this.len++] = Math.max(-1, Math.min(1, this.sum / this.n)) * 32767;
      this.sum = this.n = 0;
      if (this.len === this.out.length) {
        this.port.postMessage(this.out.buffer, [this.out.buffer]);
        this.out = new Int16Array(512);
        this.len = 0;
      }
    }
    return true;
  }
}

registerProcessor("mic", Mic);
