let webcamStream = null;
let isWebcamActive = false;
let frameInterval = null;
let lastFrameTime = performance.now();
let fpsCount = 0;

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

  event.target.classList.add('active');
  document.getElementById(`tab-${tabId}`).classList.add('active');

  if (tabId === 'benchmarks') {
    loadBenchmarkMetrics();
  }
}

async function onModelChange(modelKey) {
  try {
    const res = await fetch('/api/switch_model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelKey })
    });
    const data = await res.json();
    if (data.status === 'success') {
      document.getElementById('topEmotionBanner').innerText = `Active Model: ${modelKey.toUpperCase()}`;
    }
  } catch (err) {
    console.error('Failed to switch model:', err);
  }
}

async function toggleWebcam() {
  const btn = document.getElementById('startCamBtn');
  const video = document.getElementById('webcamVideo');
  const canvas = document.getElementById('webcamCanvas');

  if (isWebcamActive) {
    // Stop webcam
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
    }
    clearInterval(frameInterval);
    isWebcamActive = false;
    btn.innerText = 'Start Camera';
    btn.classList.remove('btn-secondary');
    document.getElementById('fpsBadge').innerText = '0 FPS';
    document.getElementById('faceCountBadge').innerText = 'Faces: 0';
    return;
  }

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } }
    });
    video.srcObject = webcamStream;
    await video.play();

    isWebcamActive = true;
    btn.innerText = 'Stop Camera';
    btn.classList.add('btn-secondary');

    // Canvas sizing
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    // Start frame loop (15 FPS max for low latency)
    frameInterval = setInterval(processWebcamFrame, 70);

  } catch (err) {
    alert('Camera access denied or unavailable: ' + err.message);
  }
}

async function processWebcamFrame() {
  if (!isWebcamActive) return;

  const video = document.getElementById('webcamVideo');
  const canvas = document.getElementById('webcamCanvas');
  const ctx = canvas.getContext('2d');

  if (video.readyState < 2) return;

  // Draw video frame to hidden canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const frameB64 = canvas.toDataURL('image/jpeg', 0.8);

  // Send to server
  const now = performance.now();
  try {
    const res = await fetch('/api/predict_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: frameB64 })
    });
    const data = await res.json();

    if (data.annotated_frame) {
      const img = new Image();
      img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      img.src = data.annotated_frame;
    }

    document.getElementById('faceCountBadge').innerText = `Faces: ${data.faces_count || 0}`;

    if (data.faces && data.faces.length > 0) {
      const topFace = data.faces[0];
      updateEmotionBars(topFace.probabilities, 'webcam');
      document.getElementById('topEmotionBanner').innerText = 
        `Detected: ${topFace.top_emotion.toUpperCase()} (${(topFace.confidence*100).toFixed(1)}%)`;
    }

    // Compute FPS
    const delta = (now - lastFrameTime) / 1000;
    lastFrameTime = now;
    fpsCount = Math.round(1 / delta);
    document.getElementById('fpsBadge').innerText = `${fpsCount} FPS`;

  } catch (err) {
    console.error('Frame process error:', err);
  }
}

function updateEmotionBars(probs, prefix = 'webcam') {
  const emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
  const prefixId = prefix === 'webcam' ? '' : 'img-';

  emotions.forEach(emo => {
    const val = probs[emo] || 0.0;
    const pctStr = (val * 100).toFixed(1) + '%';
    
    const valEl = document.getElementById(`${prefixId}val-${emo}`);
    const barEl = document.getElementById(`${prefixId}bar-${emo}`);

    if (valEl) valEl.innerText = pctStr;
    if (barEl) barEl.style.width = pctStr;
  });
}

// Image Classifier Upload Handler
async function handleFileSelect(evt) {
  const file = evt.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/predict_image', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();

    if (data.status === 'success') {
      document.getElementById('imagePreviewContainer').style.display = 'flex';
      document.getElementById('annotatedImagePreview').src = data.annotated_image;

      if (data.faces && data.faces.length > 0) {
        const topFace = data.faces[0];
        document.getElementById('imgTopEmotionCard').style.display = 'block';
        document.getElementById('imgTopEmotionText').innerText = 
          `${topFace.top_emotion.toUpperCase()} (${(topFace.confidence * 100).toFixed(1)}%)`;

        updateEmotionBars(topFace.probabilities, 'image');
      } else {
        document.getElementById('imgTopEmotionCard').style.display = 'block';
        document.getElementById('imgTopEmotionText').innerText = "No Face Detected";
      }
    }
  } catch (err) {
    alert("Image processing failed: " + err.message);
  }
}

// Drag and drop setup
const dropzone = document.getElementById('dropzone');
if (dropzone) {
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      document.getElementById('fileInput').files = files;
      handleFileSelect({ target: { files: files } });
    }
  });
}

async function loadBenchmarkMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();

    const activeModelKey = document.getElementById('modelSelect').value;
    let modelData = data[activeModelKey === 'custom' ? 'custom_cnn' : 'transfer_resnet18'] || data['custom_cnn_eval'];

    if (modelData && modelData.metrics) {
      const m = modelData.metrics;
      document.getElementById('metricAcc').innerText = `${(m.accuracy * 100).toFixed(1)}%`;
      document.getElementById('metricF1').innerText = `${m.macro_f1.toFixed(3)}`;
      document.getElementById('metricPrec').innerText = `${(m.weighted_precision * 100).toFixed(1)}%`;
      document.getElementById('metricRec').innerText = `${(m.weighted_recall * 100).toFixed(1)}%`;

      const tag = modelData.save_tag || (activeModelKey === 'custom' ? 'custom_cnn' : 'transfer_resnet18');
      document.getElementById('cmImage').src = `/models/confusion_matrix_${tag}.png?t=${Date.now()}`;
    }
  } catch (err) {
    console.error('Failed to load metrics:', err);
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch('/api/config');
    const data = await res.json();
    if (data.active_model) {
      const select = document.getElementById('modelSelect');
      if (select) {
        select.value = data.active_model;
        const banner = document.getElementById('topEmotionBanner');
        if (banner) {
          banner.innerText = `Active Model: ${data.active_model.toUpperCase()}`;
        }
      }
    }
  } catch (err) {
    console.error('Failed to fetch initial app config:', err);
  }
});

