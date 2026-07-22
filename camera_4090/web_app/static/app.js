const STORAGE_KEY = "local-llm-chat-messages";

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const inputEl = document.querySelector("#messageInput");
const imageInput = document.querySelector("#imageInput");
const imageButton = document.querySelector("#imageButton");
const sendButton = document.querySelector("#sendButton");
const clearButton = document.querySelector("#clearButton");
const clearImagesButton = document.querySelector("#clearImagesButton");
const checkButton = document.querySelector("#checkButton");
const cameraButton = document.querySelector("#cameraButton");
const statusDot = document.querySelector("#statusDot");
const statusText = document.querySelector("#statusText");
const imageTray = document.querySelector("#imageTray");
const imagePreviewList = document.querySelector("#imagePreviewList");

let messages = loadMessages();
let uploadedImages = [];
let planTimer = null;
let runWatchTimer = null;

renderMessages();
checkConnection();

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendRobotPlan();
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

inputEl.addEventListener("input", resizeInput);

clearButton.addEventListener("click", () => {
  messages = [];
  saveMessages();
  renderMessages();
});

checkButton.addEventListener("click", checkConnection);
cameraButton.addEventListener("click", checkCameras);
imageButton.addEventListener("click", () => imageInput.click());
imageInput.addEventListener("change", handleImageSelection);
clearImagesButton.addEventListener("click", clearUploadedImages);

async function sendRobotPlan() {
  const command = inputEl.value.trim();
  if (!command) {
    return;
  }

  messages.push({ role: "user", content: command });
  saveMessages();
  renderMessages();
  inputEl.value = "";
  resizeInput();

  setBusy(true, "plan");
  startPlanTimer(uploadedImages.length);
  try {
    const response = await fetch("/api/robot/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command, images: uploadedImages.map((image) => image.dataUrl) }),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }

    messages.push({
      role: "assistant",
      kind: "robot-plan",
      content: formatRobotPlan(data),
    });
    saveMessages();
    renderMessages();
    if (data.mode === "orchestrating" && data.run_id) {
      setStatus(true, "VLM contract is ready; VLA runner is processing steps");
      watchOrchestratedRun(data.run_id);
    } else if (data.vla_called) {
      setStatus(true, "VLM план отправлен в VLA");
    } else {
      setStatus(true, data.image_source === "upload" ? "план по картинке готов" : "VLM план готов");
    }
  } catch (error) {
    showError(`Не удалось построить план робота: ${error.message}`);
    setStatus(false, "offline");
  } finally {
    stopPlanTimer();
    setBusy(false);
    inputEl.focus();
  }
}

function watchOrchestratedRun(runId) {
  if (runWatchTimer !== null) {
    window.clearInterval(runWatchTimer);
  }

  const update = async () => {
    try {
      const response = await fetch(`/api/robot/runs/${encodeURIComponent(runId)}`);
      const run = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(run.detail || `HTTP ${response.status}`);
      }

      const completed = Array.isArray(run.completed_actions) ? run.completed_actions.length : 0;
      setStatus(true, `run ${run.status}; completed steps: ${completed}`);
      if (!["completed", "failed", "cancelled", "rejected"].includes(run.status)) {
        return;
      }

      window.clearInterval(runWatchTimer);
      runWatchTimer = null;
      const detail = run.error ? `: ${run.error}` : "";
      messages.push({
        role: "assistant",
        kind: "robot-plan",
        content: `orchestrator ${run.status}${detail}`,
      });
      saveMessages();
      renderMessages();
      setStatus(run.status === "completed", `run ${run.status}${detail}`);
    } catch (error) {
      if (runWatchTimer !== null) {
        window.clearInterval(runWatchTimer);
        runWatchTimer = null;
      }
      setStatus(false, `orchestrator offline: ${error.message}`);
      showError(`Orchestrator status failed: ${error.message}`);
    }
  };

  update();
  runWatchTimer = window.setInterval(update, 1500);
}

function startPlanTimer(imageCount) {
  const startedAt = Date.now();
  const source = imageCount ? `images: ${imageCount}` : "camera frame";
  const update = () => {
    const seconds = Math.floor((Date.now() - startedAt) / 1000);
    setStatus(true, `VLM is building the plan, ${source}, waiting ${seconds}s`);
  };
  update();
  planTimer = window.setInterval(update, 1000);
}

function stopPlanTimer() {
  if (planTimer !== null) {
    window.clearInterval(planTimer);
    planTimer = null;
  }
}

async function handleImageSelection() {
  const files = Array.from(imageInput.files || []);
  const imageFiles = files.filter((file) => file.type.startsWith("image/"));
  if (!imageFiles.length) {
    return;
  }

  const loaded = await Promise.all(
    imageFiles.map(async (file) => ({
      name: file.name,
      dataUrl: await readFileAsDataUrl(file),
    }))
  );

  uploadedImages = [...uploadedImages, ...loaded].slice(0, 4);
  renderImagePreviews();
  setStatus(true, `прикреплено: ${uploadedImages.length}`);
  imageInput.value = "";
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(reader.result));
    reader.addEventListener("error", () => reject(reader.error || new Error("Не удалось прочитать файл")));
    reader.readAsDataURL(file);
  });
}

function clearUploadedImages() {
  uploadedImages = [];
  renderImagePreviews();
}

function renderImagePreviews() {
  imagePreviewList.innerHTML = "";
  imageTray.classList.toggle("hidden", uploadedImages.length === 0);

  for (const image of uploadedImages) {
    const item = document.createElement("div");
    item.className = "image-preview";

    const img = document.createElement("img");
    img.src = image.dataUrl;
    img.alt = image.name;

    const label = document.createElement("span");
    label.textContent = image.name;

    item.appendChild(img);
    item.appendChild(label);
    imagePreviewList.appendChild(item);
  }
}

async function checkConnection() {
  checkButton.disabled = true;
  try {
    const response = await fetch("/api/models");
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }
    const count = Array.isArray(data.data) ? data.data.length : 0;
    setStatus(true, count ? `online, моделей: ${count}` : "online");
  } catch (error) {
    setStatus(false, `offline: ${error.message}`);
    showError(`Проверка подключения не прошла: ${error.message}`);
  } finally {
    checkButton.disabled = false;
  }
}

async function checkCameras() {
  cameraButton.disabled = true;
  try {
    const response = await fetch("/api/camera/health");
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }

    const cameras = data.cameras || {};
    const ready = Object.entries(cameras)
      .filter(([, camera]) => camera && camera.ready)
      .map(([name, camera]) => `${name} ${camera.age_ms ?? "?"} ms`);
    const missing = Object.entries(cameras)
      .filter(([, camera]) => !camera || !camera.ready)
      .map(([name]) => name);

    if (ready.length) {
      setStatus(true, `камеры: ${ready.join(", ")}`);
    } else {
      setStatus(false, "камеры не готовы");
    }

    if (missing.length) {
      showError(`Камеры ещё не готовы: ${missing.join(", ")}`);
    }
  } catch (error) {
    setStatus(false, `камеры offline: ${error.message}`);
    showError(`Проверка камер не прошла: ${error.message}`);
  } finally {
    cameraButton.disabled = false;
  }
}

function renderMessages() {
  messagesEl.innerHTML = "";

  if (messages.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "";
    messagesEl.appendChild(empty);
    return;
  }

  for (const message of messages) {
    const item = document.createElement("div");
    item.className = `message ${message.role === "user" ? "user" : "assistant"} ${message.kind === "robot-plan" ? "robot-plan" : ""}`;
    item.textContent = message.content;
    messagesEl.appendChild(item);
  }

  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showError(text) {
  const item = document.createElement("div");
  item.className = "message error";
  item.textContent = text;
  messagesEl.appendChild(item);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setStatus(isOnline, text) {
  statusDot.classList.toggle("online", isOnline);
  statusDot.classList.toggle("offline", !isOnline);
  statusText.textContent = text;
}

function setBusy(isBusy, mode = "chat") {
  sendButton.disabled = isBusy;
  imageButton.disabled = isBusy;
  inputEl.disabled = isBusy;
  sendButton.textContent = isBusy ? "…" : "➜";
}

function loadMessages() {
  try {
    const saved = sessionStorage.getItem(STORAGE_KEY);
    const parsed = saved ? JSON.parse(saved) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveMessages() {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
}

function resizeInput() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${inputEl.scrollHeight}px`;
}

function formatRobotPlan(data) {
  const steps = Array.isArray(data.plan) ? data.plan : [];
  if (data.task_feasible === false) {
    const reason = data.failure_reason || "task is not feasible from the supplied image";
    return `stop\n\nreason: ${reason}`;
  }
  if (!steps.length) {
    return "stop";
  }

  return steps
    .map((step) => {
      const action = step.action || "stop";
      const verification = step.verification;
      if (!verification) {
        return action;
      }
      const success = (verification.success || []).map(formatCondition).join(", ");
      const failure = (verification.failure || []).map(formatCondition).join(", ");
      return [action, success ? `  success: ${success}` : "", failure ? `  fail: ${failure}` : ""]
        .filter(Boolean)
        .join("\n");
    })
    .join("\n\n");
}

function formatCondition(condition) {
  if (!condition || !condition.predicate) {
    return "invalid";
  }
  const references = [condition.object, condition.target].filter(Boolean);
  return references.length ? `${condition.predicate}(${references.join(", ")})` : condition.predicate;
}
