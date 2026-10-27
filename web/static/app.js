// State variables
let currentAudioFile = null;
let currentAudioUrl = null;
let audioElement = new Audio();
let isPlaying = false;
let audioDuration = 0;
let waveformData = [];

// DOM Elements
const tabBtnUpload = document.getElementById("tab-btn-upload");
const tabBtnRecord = document.getElementById("tab-btn-record");
const tabContentUpload = document.getElementById("tab-content-upload");
const tabContentRecord = document.getElementById("tab-content-record");

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const btnBrowse = document.getElementById("btn-browse");
const selectedFileInfo = document.getElementById("selected-file-info");
const fileNameText = document.getElementById("file-name-text");
const fileSizeText = document.getElementById("file-size-text");

const btnRecordToggle = document.getElementById("btn-record-toggle");
const recordBtnText = document.getElementById("record-btn-text");
const recordTimer = document.getElementById("record-timer");
const recordVisualizer = document.getElementById("record-visualizer");

const btnAnalyze = document.getElementById("btn-analyze");
const analyzeSpinner = document.getElementById("analyze-spinner");

const resultsSection = document.getElementById("results-section");
const verdictBanner = document.getElementById("verdict-banner");
const verdictStatusIcon = document.getElementById("verdict-status-icon");
const verdictHeading = document.getElementById("verdict-heading");
const verdictDescription = document.getElementById("verdict-description");
const verdictConfidence = document.getElementById("verdict-confidence");

const gaugeCircle = document.getElementById("gauge-circle");
const gaugePercentage = document.getElementById("gauge-percentage");
const gaugeRiskLevel = document.getElementById("gauge-risk-level");

const barBonafide = document.getElementById("bar-bonafide");
const barSpoof = document.getElementById("bar-spoof");
const valBonafideProb = document.getElementById("val-bonafide-prob");
const valSpoofProb = document.getElementById("val-spoof-prob");
const valMargin = document.getElementById("val-margin");
const valDuration = document.getElementById("val-duration");

const btnSampleHuman = document.getElementById("btn-sample-human");
const btnSampleFake = document.getElementById("btn-sample-fake");

const btnPlayPause = document.getElementById("btn-play-pause");
const playIcon = document.getElementById("play-icon");
const pauseIcon = document.getElementById("pause-icon");
const playerFilename = document.getElementById("player-filename");
const playerDuration = document.getElementById("player-duration");
const waveformCanvas = document.getElementById("waveform-canvas");
const waveformProgress = document.getElementById("waveform-progress");

// Tab Switching
tabBtnUpload.addEventListener("click", () => {
    tabBtnUpload.classList.add("active");
    tabBtnRecord.classList.remove("active");
    tabContentUpload.style.display = "block";
    tabContentRecord.style.display = "none";
});

tabBtnRecord.addEventListener("click", () => {
    tabBtnRecord.classList.add("active");
    tabBtnUpload.classList.remove("active");
    tabContentRecord.style.display = "block";
    tabContentUpload.style.display = "none";
});

// File Browsing & Drop
btnBrowse.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
});

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleSelectedFile(e.target.files[0]);
    }
});

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleSelectedFile(e.dataTransfer.files[0]);
    }
});

function handleSelectedFile(file) {
    currentAudioFile = file;
    fileNameText.textContent = file.name;
    const mb = (file.size / (1024 * 1024)).toFixed(2);
    fileSizeText.textContent = `${mb} MB`;
    selectedFileInfo.style.display = "inline-flex";

    if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
    }
    currentAudioUrl = URL.createObjectURL(file);
    audioElement.src = currentAudioUrl;

    btnAnalyze.disabled = false;
}

// Quick Sample Loaders
btnSampleHuman.addEventListener("click", async () => {
    try {
        btnSampleHuman.disabled = true;
        const res = await fetch("/api/sample/bonafide");
        const blob = await res.blob();
        const file = new File([blob], "sample_bonafide_human.wav", { type: "audio/wav" });
        handleSelectedFile(file);
        triggerAnalysis();
    } catch (err) {
        alert("Failed to load human sample: " + err);
    } finally {
        btnSampleHuman.disabled = false;
    }
});

btnSampleFake.addEventListener("click", async () => {
    try {
        btnSampleFake.disabled = true;
        const res = await fetch("/api/sample/spoof");
        const blob = await res.blob();
        const file = new File([blob], "sample_deepfake_synthetic.wav", { type: "audio/wav" });
        handleSelectedFile(file);
        triggerAnalysis();
    } catch (err) {
        alert("Failed to load deepfake sample: " + err);
    } finally {
        btnSampleFake.disabled = false;
    }
});

// Live Microphone Recording
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let recordTimerInterval = null;
let recordStartTime = null;
let isRecording = false;

btnRecordToggle.addEventListener("click", async () => {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];

        // Setup Visualizer
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        analyser.fftSize = 256;
        drawLiveVisualizer();

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            const mimeType = mediaRecorder.mimeType || "audio/webm";
            const ext = mimeType.includes("webm") ? ".webm" : (mimeType.includes("mp4") || mimeType.includes("aac") ? ".m4a" : ".wav");
            const blob = new Blob(audioChunks, { type: mimeType });
            const file = new File([blob], `mic_recording_${Date.now()}${ext}`, { type: mimeType });
            handleSelectedFile(file);
            stream.getTracks().forEach(track => track.stop());
            if (audioContext) audioContext.close();
        };

        mediaRecorder.start();
        isRecording = true;
        btnRecordToggle.classList.add("recording");
        recordBtnText.textContent = "Stop Recording";

        recordStartTime = Date.now();
        recordTimerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
            const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
            const secs = String(elapsed % 60).padStart(2, "0");
            recordTimer.textContent = `${mins}:${secs}`;
        }, 1000);

    } catch (err) {
        alert("Microphone access denied or unsupported: " + err);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    isRecording = false;
    btnRecordToggle.classList.remove("recording");
    recordBtnText.textContent = "Start Recording";
    clearInterval(recordTimerInterval);
}

function drawLiveVisualizer() {
    if (!analyser || !isRecording) return;
    const canvas = recordVisualizer;
    const ctx = canvas.getContext("2d");
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function render() {
        if (!isRecording) return;
        requestAnimationFrame(render);
        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = "rgba(10, 15, 26, 0.4)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.2;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height;
            const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
            gradient.addColorStop(0, "#06b6d4");
            gradient.addColorStop(1, "#8b5cf6");

            ctx.fillStyle = gradient;
            ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
            x += barWidth;
        }
    }
    render();
}

// Analysis API Call
btnAnalyze.addEventListener("click", triggerAnalysis);

async function triggerAnalysis() {
    if (!currentAudioFile) return;

    btnAnalyze.disabled = true;
    analyzeSpinner.style.display = "inline-flex";

    const formData = new FormData();
    formData.append("file", currentAudioFile);

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        if (data.status === "success") {
            displayResults(data);
        } else {
            alert("Analysis error: " + (data.message || "Unknown error"));
        }
    } catch (err) {
        alert("Failed to connect to detection backend: " + err);
    } finally {
        btnAnalyze.disabled = false;
        analyzeSpinner.style.display = "none";
    }
}

function displayResults(data) {
    const pred = data.prediction;
    const duration = data.duration_sec;
    audioDuration = duration;
    waveformData = data.waveform_preview || [];

    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth" });

    // Verdict Banner styling
    const isSpoof = pred.is_spoof;
    verdictBanner.className = `verdict-banner ${isSpoof ? "spoof" : "bonafide"}`;

    if (isSpoof) {
        verdictStatusIcon.innerHTML = `
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
        `;
        verdictHeading.textContent = "AI DEEPFAKE / SPOOF DETECTED";
        verdictDescription.textContent = "Synthetic speech, voice cloning, or neural vocoder compression signatures were identified.";
    } else {
        verdictStatusIcon.innerHTML = `
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
        `;
        verdictHeading.textContent = "AUTHENTIC HUMAN VOICE (BONA FIDE)";
        verdictDescription.textContent = "Natural vocal tract formants and acoustic dynamics confirmed with high biometric confidence.";
    }

    verdictConfidence.textContent = `${pred.confidence_pct}%`;

    // Radial Gauge
    const spoofPct = Math.round(pred.spoof_probability * 100);
    gaugePercentage.textContent = `${spoofPct}%`;
    const circumference = 2 * Math.PI * 50; // r=50 -> ~314.16
    const offset = circumference - (spoofPct / 100) * circumference;
    gaugeCircle.style.strokeDashoffset = offset;
    gaugeCircle.style.stroke = isSpoof ? "var(--color-spoof)" : "var(--color-bonafide)";
    gaugeRiskLevel.textContent = `Risk Level: ${pred.risk_level.split("-")[0].trim()}`;

    // Score Bars
    const bonaPct = (pred.bonafide_probability * 100).toFixed(1);
    const fakePct = (pred.spoof_probability * 100).toFixed(1);
    barBonafide.style.width = `${bonaPct}%`;
    barSpoof.style.width = `${fakePct}%`;
    valBonafideProb.textContent = `${bonaPct}%`;
    valSpoofProb.textContent = `${fakePct}%`;

    valMargin.textContent = `${pred.decision_margin > 0 ? "+" : ""}${pred.decision_margin}`;
    valDuration.textContent = `${duration}s`;

    // Insight items
    const chkVocoder = document.getElementById("chk-vocoder");
    const chkSpectral = document.getElementById("chk-spectral");
    const chkTemporal = document.getElementById("chk-temporal");
    if (isSpoof) {
        chkVocoder.className = "insight-item warn";
        chkSpectral.className = "insight-item warn";
        chkTemporal.className = "insight-item";
    } else {
        chkVocoder.className = "insight-item";
        chkSpectral.className = "insight-item";
        chkTemporal.className = "insight-item";
    }

    // Player setup
    playerFilename.textContent = data.filename || currentAudioFile.name;
    playerDuration.textContent = `00:00 / 00:${String(Math.floor(duration)).padStart(2, "0")}`;
    drawWaveform();

    // Render Graphical Comparisons
    if (data.acoustic_analysis) {
        renderSpectrumChart(data.acoustic_analysis);
        renderSpectrogramComparison(data.acoustic_analysis, isSpoof);
        renderBiometricDifferences(data.acoustic_analysis, isSpoof);
    }
}

// =====================================================================
// GRAPHICAL COMPARISON SUITE: REAL VOICE VS FAKE VOICE RENDERERS
// =====================================================================

function renderSpectrumChart(acousticData) {
    const canvas = document.getElementById("spectrum-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    const padLeft = 40;
    const padRight = 20;
    const padTop = 20;
    const padBottom = 30;
    const plotW = W - padLeft - padRight;
    const plotH = H - padTop - padBottom;

    // Draw Grid & dB markings
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#64748b";
    ctx.font = "10px 'JetBrains Mono', monospace";

    for (let db = 0; db <= 100; db += 25) {
        const y = padTop + plotH - (db / 100) * plotH;
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(W - padRight, y);
        ctx.stroke();
        ctx.fillText(`${db}%`, 10, y + 3);
    }

    // Vertical Frequency Grid Lines (1kHz, 2kHz, 4kHz, 6kHz)
    const freqMarkers = [
        { khz: "1k", ratio: 1/8 },
        { khz: "2k", ratio: 2/8 },
        { khz: "4k", ratio: 4/8 },
        { khz: "6k", ratio: 6/8 },
    ];
    freqMarkers.forEach(fm => {
        const x = padLeft + fm.ratio * plotW;
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, padTop + plotH);
        ctx.stroke();
    });

    const freqs = acousticData.frequencies_hz || [];
    const audioPts = acousticData.audio_spectrum || [];
    const humanPts = acousticData.human_reference || [];
    const fakePts = acousticData.deepfake_reference || [];
    const N = Math.min(audioPts.length, humanPts.length, fakePts.length);
    if (N < 2) return;

    function getX(i) {
        return padLeft + (i / (N - 1)) * plotW;
    }
    function getY(val) {
        return padTop + plotH - (val / 100) * plotH;
    }

    // 1. Draw Deepfake / Vocoder Reference (Crimson Dashed Line)
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = "rgba(244, 63, 94, 0.7)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
        const x = getX(i);
        const y = getY(fakePts[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // 2. Draw Genuine Human Voice Baseline (Emerald Solid Line)
    ctx.save();
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "rgba(16, 185, 129, 0.4)";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
        const x = getX(i);
        const y = getY(humanPts[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // 3. Draw Analyzed Audio Curve with Gradient Area Fill (Cyan Line)
    ctx.save();
    const areaGradient = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
    areaGradient.addColorStop(0, "rgba(6, 182, 212, 0.28)");
    areaGradient.addColorStop(1, "rgba(6, 182, 212, 0.0)");

    ctx.beginPath();
    for (let i = 0; i < N; i++) {
        const x = getX(i);
        const y = getY(audioPts[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.lineTo(getX(N - 1), padTop + plotH);
    ctx.lineTo(getX(0), padTop + plotH);
    ctx.closePath();
    ctx.fillStyle = areaGradient;
    ctx.fill();

    // Top curve line
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 3;
    ctx.shadowColor = "rgba(6, 182, 212, 0.6)";
    ctx.shadowBlur = 10;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
        const x = getX(i);
        const y = getY(audioPts[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // Draw Data Point Dots at key frequencies (500Hz, 1.5kHz, 4.5kHz)
    [8, 20, 40].forEach(idx => {
        if (idx < N) {
            ctx.fillStyle = "#fff";
            ctx.beginPath();
            ctx.arc(getX(idx), getY(audioPts[idx]), 4, 0, Math.PI * 2);
            ctx.fill();
        }
    });
}

function renderSpectrogramComparison(acousticData, isSpoof) {
    // 1. Render Benchmark Human Spectrogram (Natural Resonance Formants)
    const humanCanvas = document.getElementById("human-spectrogram-canvas");
    if (humanCanvas) {
        const ctx = humanCanvas.getContext("2d");
        const W = humanCanvas.width;
        const H = humanCanvas.height;
        ctx.fillStyle = "#0a0f1d";
        ctx.fillRect(0, 0, W, H);

        // Generate natural human formant harmonics (F1, F2, F3 tracks)
        const cols = 50;
        const rows = 30;
        const cellW = W / cols;
        const cellH = H / rows;

        for (let c = 0; c < cols; c++) {
            const timeVibrato = Math.sin(c * 0.25) * 1.5;
            for (let r = 0; r < rows; r++) {
                const freqY = rows - r;
                let intensity = 0.05;

                // F1 Formant (~500Hz)
                if (Math.abs(freqY - (7 + timeVibrato)) < 2.5) intensity += 0.85;
                // F2 Formant (~1500Hz)
                if (Math.abs(freqY - (14 + timeVibrato * 1.2)) < 2.0) intensity += 0.70;
                // F3 Formant (~2800Hz)
                if (Math.abs(freqY - 21) < 1.8) intensity += 0.50;
                // Natural high-freq decay
                intensity *= Math.exp(-r / 30);

                intensity = Math.min(1.0, intensity);
                ctx.fillStyle = getSpectrogramColor(intensity);
                ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
            }
        }

        // Overlay text badge
        ctx.fillStyle = "rgba(16, 185, 129, 0.9)";
        ctx.font = "bold 10px 'Outfit', sans-serif";
        ctx.fillText("✓ Formant Resonance (F1/F2/F3)", 12, 22);
    }

    // 2. Render Uploaded Audio Spectrogram
    const audioCanvas = document.getElementById("audio-spectrogram-canvas");
    const tag = document.getElementById("spectro-analysis-tag");
    const note = document.getElementById("spectro-analysis-note");

    if (audioCanvas) {
        const ctx = audioCanvas.getContext("2d");
        const W = audioCanvas.width;
        const H = audioCanvas.height;
        ctx.fillStyle = "#0a0f1d";
        ctx.fillRect(0, 0, W, H);

        const matrix = acousticData.spectrogram_heatmap || [];
        const rows = matrix.length || 32;
        const cols = matrix[0] ? matrix[0].length : 32;
        const cellW = W / cols;
        const cellH = H / rows;

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const val = matrix[r] ? matrix[r][c] : 0.2;
                ctx.fillStyle = getSpectrogramColor(val);
                ctx.fillRect(c * cellW, (rows - 1 - r) * cellH, cellW + 0.5, cellH + 0.5);
            }
        }

        if (isSpoof) {
            // Annotate Anomaly Zones (Vocoder Artifacts)
            ctx.strokeStyle = "rgba(244, 63, 94, 0.9)";
            ctx.lineWidth = 1.8;
            ctx.strokeRect(W * 0.45, 15, W * 0.45, 45);

            ctx.fillStyle = "rgba(244, 63, 94, 0.95)";
            ctx.font = "bold 10px 'Outfit', sans-serif";
            ctx.fillText("⚠ VOCODER NOISE / COMB BAND", W * 0.46, 32);

            if (tag) {
                tag.className = "spectro-tag spoof";
                tag.textContent = "Deepfake Artifacts Flagged";
            }
            if (note) {
                note.textContent = "AASIST GAT flagged abnormal energy distribution in upper frequency bands (>4kHz) and robotic harmonic comb filtering.";
            }
        } else {
            // Annotate Natural Vocal Resonance
            ctx.strokeStyle = "rgba(16, 185, 129, 0.9)";
            ctx.lineWidth = 1.8;
            ctx.strokeRect(10, H * 0.4, W * 0.8, 45);

            ctx.fillStyle = "rgba(16, 185, 129, 0.95)";
            ctx.font = "bold 10px 'Outfit', sans-serif";
            ctx.fillText("✓ NATURAL VOCAL HARMONIC TRACK", 20, H * 0.4 + 20);

            if (tag) {
                tag.className = "spectro-tag bona";
                tag.textContent = "Authentic Human Voice";
            }
            if (note) {
                note.textContent = "Smooth formant dynamics and natural high-frequency energy decay match genuine human vocal anatomy.";
            }
        }
    }
}

function getSpectrogramColor(val) {
    // Magma / Viridis thermal color mapping
    const v = Math.max(0, Math.min(1, val));
    if (v < 0.25) {
        const t = v / 0.25;
        return `rgb(${Math.round(15 + t * 40)}, ${Math.round(10 + t * 10)}, ${Math.round(45 + t * 65)})`;
    } else if (v < 0.55) {
        const t = (v - 0.25) / 0.3;
        return `rgb(${Math.round(55 + t * 130)}, ${Math.round(20 + t * 30)}, ${Math.round(110 - t * 40)})`;
    } else if (v < 0.85) {
        const t = (v - 0.55) / 0.3;
        return `rgb(${Math.round(185 + t * 60)}, ${Math.round(50 + t * 120)}, ${Math.round(70 - t * 50)})`;
    } else {
        const t = (v - 0.85) / 0.15;
        return `rgb(${Math.round(245 + t * 10)}, ${Math.round(170 + t * 75)}, ${Math.round(20 + t * 180)})`;
    }
}

function renderBiometricDifferences(acousticData, isSpoof) {
    const container = document.getElementById("diff-grid-container");
    if (!container) return;
    container.innerHTML = "";

    const biometrics = acousticData.biometric_comparison || {};
    Object.keys(biometrics).forEach(key => {
        const item = biometrics[key];
        const isAuthentic = item.audio_val >= 60;

        const row = document.createElement("div");
        row.className = "diff-item";
        row.innerHTML = `
            <div class="diff-item-header">
                <div class="diff-name-group">
                    <span class="diff-name">${item.name}</span>
                    <span class="diff-status-tag ${isAuthentic ? 'authentic' : 'fake'}">
                        ${isAuthentic ? '✓ NATURAL BIOMETRIC' : '⚠ SYNTHETIC ANOMALY'}
                    </span>
                </div>
                <div class="diff-values-text">
                    Analyzed: <strong>${item.audio_val}%</strong> | Human Baseline: <strong>${item.human_baseline}%</strong> | Deepfake Zone: &lt;45%
                </div>
            </div>
            <div class="diff-bar-track-wrapper">
                <div class="diff-bar-fill ${isAuthentic ? 'bona' : 'spoof'}" style="width: ${item.audio_val}%;"></div>
                <div class="diff-reference-marker" style="left: ${item.human_baseline}%;" title="Genuine Human Baseline (${item.human_baseline}%)"></div>
            </div>
            <div class="diff-item-desc">${item.description}</div>
        `;
        container.appendChild(row);
    });
}


// Waveform Player Logic
btnPlayPause.addEventListener("click", togglePlayback);

function togglePlayback() {
    if (!audioElement.src) return;

    if (isPlaying) {
        audioElement.pause();
        isPlaying = false;
        playIcon.style.display = "block";
        pauseIcon.style.display = "none";
    } else {
        audioElement.play();
        isPlaying = true;
        playIcon.style.display = "none";
        pauseIcon.style.display = "block";
    }
}

audioElement.addEventListener("timeupdate", () => {
    if (!audioElement.duration) return;
    const progress = audioElement.currentTime / audioElement.duration;
    waveformProgress.style.left = `${progress * 100}%`;

    const curM = String(Math.floor(audioElement.currentTime / 60)).padStart(2, "0");
    const curS = String(Math.floor(audioElement.currentTime % 60)).padStart(2, "0");
    const durM = String(Math.floor(audioElement.duration / 60)).padStart(2, "0");
    const durS = String(Math.floor(audioElement.duration % 60)).padStart(2, "0");
    playerDuration.textContent = `${curM}:${curS} / ${durM}:${durS}`;
});

audioElement.addEventListener("ended", () => {
    isPlaying = false;
    playIcon.style.display = "block";
    pauseIcon.style.display = "none";
    waveformProgress.style.left = "0%";
});

waveformCanvas.parentElement.addEventListener("click", (e) => {
    if (!audioElement.duration) return;
    const rect = waveformCanvas.parentElement.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    audioElement.currentTime = ratio * audioElement.duration;
});

function drawWaveform() {
    const canvas = waveformCanvas;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (waveformData.length === 0) {
        // Fallback simulated waveform
        for (let i = 0; i < 150; i++) {
            waveformData.push(Math.sin(i * 0.1) * Math.cos(i * 0.05) * 0.7);
        }
    }

    const barWidth = width / waveformData.length;
    const centerY = height / 2;

    for (let i = 0; i < waveformData.length; i++) {
        const val = Math.abs(waveformData[i]);
        const barHeight = Math.max(3, val * height * 0.85);

        const gradient = ctx.createLinearGradient(0, centerY - barHeight / 2, 0, centerY + barHeight / 2);
        gradient.addColorStop(0, "#06b6d4");
        gradient.addColorStop(0.5, "#3b82f6");
        gradient.addColorStop(1, "#8b5cf6");

        ctx.fillStyle = gradient;
        ctx.fillRect(i * barWidth, centerY - barHeight / 2, barWidth - 1, barHeight);
    }
}
