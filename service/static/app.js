const recordBtn = document.getElementById("recordBtn");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const verdictEl = document.getElementById("verdict");
const confidenceEl = document.getElementById("confidence");
const modelVersionEl = document.getElementById("modelVersion");
const latencyEl = document.getElementById("latency");

let mediaRecorder = null;
let audioChunks = [];
let mediaStream = null;
let isRecording = false;

recordBtn.addEventListener("click", () => {
  isRecording ? stopRecording() : startRecording();
});

async function startRecording() {
  errorEl.classList.remove("visible");
  resultEl.classList.remove("visible");

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    showError("microphone access denied: " + err.message);
    return;
  }

  audioChunks = [];
  mediaRecorder = new MediaRecorder(mediaStream);
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) audioChunks.push(e.data);
  };
  mediaRecorder.onstop = handleRecordingStop;
  mediaRecorder.start();

  isRecording = true;
  recordBtn.classList.add("recording");
  statusEl.textContent = "recording… press to stop";
}

function stopRecording() {
  mediaRecorder.stop();
  mediaStream.getTracks().forEach((track) => track.stop());

  isRecording = false;
  recordBtn.classList.remove("recording");
  recordBtn.disabled = true;
  statusEl.textContent = "processing…";
}

// The backend reads WAV/FLAC via libsndfile, but MediaRecorder only produces
// WebM/Opus (or MP4/AAC on Safari) -- decode via Web Audio API and re-encode as
// a plain 16-bit PCM WAV client-side rather than adding a server-side transcode step.
function audioBufferToWav(buffer) {
  const numChannels = 1;
  const sampleRate = buffer.sampleRate;
  const length = buffer.length;

  const mono = new Float32Array(length);
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const channelData = buffer.getChannelData(ch);
    for (let i = 0; i < length; i++) mono[i] += channelData[i] / buffer.numberOfChannels;
  }

  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = length * blockAlign;
  const arrayBuffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(arrayBuffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < length; i++) {
    const clamped = Math.max(-1, Math.min(1, mono[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([arrayBuffer], { type: "audio/wav" });
}

async function handleRecordingStop() {
  try {
    const recordedBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
    const arrayBuffer = await recordedBlob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
    const wavBlob = audioBufferToWav(decoded);
    await audioCtx.close();
    await sendForDetection(wavBlob);
  } catch (err) {
    showError("could not process recording: " + err.message);
    resetButton();
  }
}

async function sendForDetection(wavBlob) {
  const formData = new FormData();
  formData.append("file", wavBlob, "recording.wav");

  try {
    const res = await fetch("/detect", { method: "POST", body: formData });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `server error (${res.status})`);
    }
    renderResult(await res.json());
  } catch (err) {
    showError(err.message);
  } finally {
    resetButton();
  }
}

function renderResult(data) {
  errorEl.classList.remove("visible");
  resultEl.classList.add("visible");
  verdictEl.textContent = data.verdict === "human" ? "HUMAN" : "AI-GENERATED";
  verdictEl.className = "verdict " + data.verdict;
  confidenceEl.textContent = `confidence ${(data.confidence * 100).toFixed(1)}%`;
  modelVersionEl.textContent = data.model_version;
  latencyEl.textContent = `${data.processing_time_ms.toFixed(0)}ms`;
}

function showError(message) {
  resultEl.classList.remove("visible");
  errorEl.textContent = message;
  errorEl.classList.add("visible");
}

function resetButton() {
  recordBtn.disabled = false;
  statusEl.textContent = "press to record";
}
