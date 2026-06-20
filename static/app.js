const fileInput = document.querySelector("#audio-file");
const predictButton = document.querySelector("#predict-button");
const resetButton = document.querySelector("#reset-button");
const fileSummary = document.querySelector("#file-summary");
const audioPreview = document.querySelector("#audio-preview");
const waveform = document.querySelector("#waveform");
const resultEmpty = document.querySelector("#result-empty");
const resultContent = document.querySelector("#result-content");
const speakerName = document.querySelector("#speaker-name");
const confidence = document.querySelector("#confidence");
const predictions = document.querySelector("#top-predictions");
const errorMessage = document.querySelector("#error-message");
const serviceStatus = document.querySelector("#service-status");

let selectedFile = null;
let previewUrl = null;

function resetResults() {
  resultContent.hidden = true;
  resultEmpty.hidden = false;
  errorMessage.hidden = true;
  errorMessage.textContent = "";
  predictions.replaceChildren();
}

function clearSelection() {
  selectedFile = null;
  fileInput.value = "";
  predictButton.disabled = true;
  resetButton.disabled = true;
  fileSummary.textContent = "No audio selected";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  audioPreview.removeAttribute("src");
  audioPreview.hidden = true;
  waveform.hidden = true;
  resetResults();
}

function formatBytes(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

async function drawWaveform(file) {
  try {
    const audioContext = new AudioContext();
    const buffer = await audioContext.decodeAudioData(await file.arrayBuffer());
    const samples = buffer.getChannelData(0);
    const context = waveform.getContext("2d");
    const { width, height } = waveform;
    const step = Math.max(1, Math.floor(samples.length / width));

    context.clearRect(0, 0, width, height);
    context.strokeStyle = "#16727c";
    context.lineWidth = 1.5;
    context.beginPath();
    for (let x = 0; x < width; x += 1) {
      let minimum = 1;
      let maximum = -1;
      const start = x * step;
      const end = Math.min(start + step, samples.length);
      for (let index = start; index < end; index += 1) {
        minimum = Math.min(minimum, samples[index]);
        maximum = Math.max(maximum, samples[index]);
      }
      context.moveTo(x, (1 + minimum) * height / 2);
      context.lineTo(x, (1 + maximum) * height / 2);
    }
    context.stroke();
    waveform.hidden = false;
    await audioContext.close();
  } catch {
    waveform.hidden = true;
  }
}

fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  if (!file) return;
  selectedFile = file;
  fileSummary.textContent = `${file.name} - ${formatBytes(file.size)}`;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  audioPreview.src = previewUrl;
  audioPreview.hidden = false;
  drawWaveform(file);
  predictButton.disabled = false;
  resetButton.disabled = false;
  resetResults();
});

resetButton.addEventListener("click", clearSelection);

predictButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  predictButton.disabled = true;
  errorMessage.hidden = true;
  predictButton.textContent = "Identifying...";

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);
    const response = await fetch("/api/predict", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Prediction failed.");

    speakerName.textContent = payload.prediction.speaker;
    confidence.textContent = `${payload.prediction.confidence.toFixed(2)}%`;
    predictions.replaceChildren(...payload.top_predictions.map((item) => {
      const row = document.createElement("li");
      row.innerHTML = `<span class="rank">${item.rank}</span><span>${item.speaker}</span><span class="probability">${item.confidence.toFixed(2)}%</span>`;
      return row;
    }));
    resultEmpty.hidden = true;
    resultContent.hidden = false;
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.hidden = false;
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Identify speaker";
  }
});

async function checkService() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    if (!response.ok || payload.status !== "ready") throw new Error();
    serviceStatus.textContent = `Ready / ${payload.speakers} speakers`;
    serviceStatus.className = "service-status ready";
  } catch {
    serviceStatus.textContent = "Service unavailable";
    serviceStatus.className = "service-status error";
  }
}

checkService();
