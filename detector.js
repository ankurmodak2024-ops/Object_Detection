const video = document.getElementById("camera");
const overlay = document.getElementById("detection-overlay");
const context = overlay.getContext("2d");
const cameraScreen = document.getElementById("camera-screen");
const cameraStatus = document.getElementById("camera-status");
const cameraCardStatus = document.getElementById("camera-card-status");
const detectionCardStatus = document.getElementById("detection-card-status");
const startButton = document.getElementById("start-camera");
const stopButton = document.getElementById("stop-camera");
const targetObjectsInput = document.getElementById("target-objects");
const confidenceInput = document.getElementById("confidence-threshold");
const confidenceValue = document.getElementById("confidence-value");

let stream = null;
let cameraIsRunning = false;
let pendingRequest = null;

function setCameraStatus(message, state) {
    cameraStatus.textContent = message;
    cameraStatus.parentElement.classList.toggle("is-active", state === "active");
    cameraStatus.parentElement.classList.toggle("is-error", state === "error");
    cameraCardStatus.textContent = state === "active" ? "Connected" : state === "error" ? "Error" : "Off";
    cameraCardStatus.classList.toggle("is-error", state === "error");
}

function drawDetections(detections) {
    context.clearRect(0, 0, overlay.width, overlay.height);
    context.lineWidth = Math.max(2, overlay.width / 400);
    context.font = `${Math.max(14, overlay.width / 45)}px Arial`;
    context.textBaseline = "top";

    for (const detection of detections) {
        const [x1, y1, x2, y2] = detection.box;
        const label = `${detection.label} ${(detection.confidence * 100).toFixed(0)}%`;
        const labelHeight = Math.max(22, overlay.width / 32);
        const labelWidth = context.measureText(label).width + 12;

        context.strokeStyle = "#26d9b1";
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);
        context.fillStyle = "#26d9b1";
        context.fillRect(x1, Math.max(0, y1 - labelHeight), labelWidth, labelHeight);
        context.fillStyle = "#07131b";
        context.fillText(label, x1 + 6, Math.max(0, y1 - labelHeight) + 4);
    }
}

async function detectNextFrame() {
    if (!cameraIsRunning || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        if (cameraIsRunning) {
            window.setTimeout(detectNextFrame, 250);
        }
        return;
    }

    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCanvas.getContext("2d").drawImage(video, 0, 0);

    try {
        const imageBlob = await new Promise((resolve, reject) => {
            captureCanvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error("Could not capture a camera frame."));
                }
            }, "image/jpeg", 0.8);
        });
        if (!cameraIsRunning) {
            return;
        }

        const formData = new FormData();
        formData.append("image", imageBlob, "camera.jpg");
        const targetClasses = targetObjectsInput.value.trim();
        if (targetClasses) {
            formData.append("classes", targetClasses);
        }
        formData.append("confidence", confidenceInput.value);
        pendingRequest = new AbortController();
        const response = await fetch("/detect", {
            method: "POST",
            body: formData,
            signal: pendingRequest.signal,
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Object detection failed.");
        }

        if (cameraIsRunning) {
            drawDetections(result.detections);
            const count = result.detections.length;
            const targetSummary = targetClasses ? ` for ${targetClasses}` : "";
            detectionCardStatus.textContent = `${count} object${count === 1 ? "" : "s"} detected${targetSummary}`;
            detectionCardStatus.classList.remove("is-error");
            setCameraStatus("Camera on", "active");
        }
    } catch (error) {
        if (error.name !== "AbortError" && cameraIsRunning) {
            setCameraStatus(`Detection failed: ${error.message}`, "error");
            detectionCardStatus.textContent = "Unavailable";
            detectionCardStatus.classList.add("is-error");
        }
    } finally {
        pendingRequest = null;
        if (cameraIsRunning) {
            window.setTimeout(detectNextFrame, 250);
        }
    }
}

confidenceInput.addEventListener("input", () => {
    confidenceValue.value = `${Math.round(Number(confidenceInput.value) * 100)}%`;
});

startButton.addEventListener("click", async () => {
    startButton.disabled = true;
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Camera access requires HTTPS or localhost in a supported browser.");
        }
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        await video.play();

        cameraIsRunning = true;
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        cameraScreen.classList.add("is-active");
        stopButton.disabled = false;
        const targetClasses = targetObjectsInput.value.trim();
        detectionCardStatus.textContent = targetClasses ? `Detecting ${targetClasses}...` : "Detecting all common objects...";
        setCameraStatus("Camera on", "active");
        detectNextFrame();
    } catch (error) {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            stream = null;
        }
        setCameraStatus(error.name === "NotAllowedError" ? "Camera permission denied" : error.message, "error");
        startButton.disabled = false;
    }
});

stopButton.addEventListener("click", () => {
    cameraIsRunning = false;
    if (pendingRequest) {
        pendingRequest.abort();
        pendingRequest = null;
    }
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }
    video.srcObject = null;
    context.clearRect(0, 0, overlay.width, overlay.height);
    cameraScreen.classList.remove("is-active");
    detectionCardStatus.textContent = "Inactive";
    detectionCardStatus.classList.remove("is-error");
    setCameraStatus("Camera off", "off");
    startButton.disabled = false;
    stopButton.disabled = true;
});
