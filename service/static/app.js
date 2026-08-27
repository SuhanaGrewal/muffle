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
