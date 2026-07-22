const screens = [...document.querySelectorAll(".screen")];
const intro = document.querySelector(".intro");
const task = document.querySelector(".task");
const working = document.querySelector(".working");
const done = document.querySelector(".done");
const form = document.querySelector("#commandForm");
const input = document.querySelector("#command");
const taskText = document.querySelector("#taskText");
const note = document.querySelector("#executionNote");
const taskHint = document.querySelector(".task-hint");
const again = document.querySelector("#again");
const approveRun = document.querySelector("#approveRun");
const stopRun = document.querySelector("#stopRun");

const POLL_INTERVAL_MS = 1250;
let currentRunId = null;
let isRunning = false;
let pollTimer = null;
let introTimer = null;
let stopRequested = false;

function show(screen) {
  screens.forEach((item) => item.classList.remove("active"));
  screen.classList.add("active");
}

function openTask() {
  if (!intro.classList.contains("active")) return;
  if (introTimer !== null) window.clearTimeout(introTimer);
  intro.style.opacity = "0";
  intro.style.pointerEvents = "none";
  show(task);
  input.focus();
}

introTimer = window.setTimeout(openTask, 1000);

function setWorking(action, status) {
  taskText.textContent = action;
  note.textContent = status;
  show(working);
}

function stopPolling() {
  if (pollTimer !== null) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function latestEvent(events, eventName) {
  if (!Array.isArray(events)) return null;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index]?.event === eventName) return events[index];
  }
  return null;
}

function currentAction(events, fallback) {
  const ready = latestEvent(events, "action_ready");
  return typeof ready?.action === "string" ? ready.action : fallback;
}

function firstPlannedAction(run) {
  const action = run.contract?.plan?.[0]?.action;
  return typeof action === "string" ? action : run.task || "Выполняю команду";
}

function updateWorkingView(run) {
  const events = run.events;
  const action = currentAction(events, run.task || "Выполняю команду");
  const lastState = Array.isArray(events)
    ? [...events].reverse().find((item) => ["action_ready", "visual_verification_started", "contract_replanned"].includes(item?.event))
    : null;

  if (lastState?.event === "visual_verification_started") {
    setWorking(action, "Проверяю результат");
    return;
  }
  if (lastState?.event === "contract_replanned") {
    setWorking(action, "Корректирую план");
    return;
  }
  if (lastState?.event === "action_ready") {
    setWorking(action, "Выполняю действие");
    return;
  }
  setWorking(action, "Строю план по камерам");
}

function terminalError(run) {
  if (run.status === "cancelled") return "Задача остановлена оператором.";
  const rejected = run.contract?.failure_reason;
  return run.error || rejected || "Задача остановлена до завершения.";
}

async function cancelCurrentRun() {
  if (!isRunning || stopRequested) return;
  stopRequested = true;
  stopRun.disabled = true;
  note.textContent = "Останавливаю задачу...";

  if (!currentRunId) return;
  try {
    const response = await fetch(`/api/robot/runs/${encodeURIComponent(currentRunId)}/cancel`, {
      method: "POST",
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || `Ошибка ${response.status}`);
    note.textContent = "Команда остановки отправлена";
  } catch (error) {
    stopRequested = false;
    stopRun.disabled = false;
    note.textContent = error.message || "Не удалось остановить задачу.";
  }
}

async function approveCurrentRun() {
  if (!isRunning || !currentRunId || approveRun.disabled) return;
  approveRun.disabled = true;
  note.textContent = "Передаю план в VLA...";
  try {
    const response = await fetch(`/api/robot/runs/${encodeURIComponent(currentRunId)}/approve`, {
      method: "POST",
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || `Ошибка ${response.status}`);
    note.textContent = "Запуск подтверждён";
  } catch (error) {
    approveRun.disabled = false;
    note.textContent = error.message || "Не удалось подтвердить запуск.";
  }
}

async function refreshRun() {
  if (!currentRunId) return;

  try {
    const response = await fetch(`/api/robot/runs/${encodeURIComponent(currentRunId)}`);
    const run = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(run.detail || `Ошибка ${response.status}`);

    if (run.status === "completed") {
      stopPolling();
      isRunning = false;
      approveRun.disabled = true;
      show(done);
      return;
    }
    if (run.status === "awaiting_approval") {
      approveRun.disabled = false;
      setWorking(firstPlannedAction(run), "План готов. Подтвердите запуск или остановите задачу.");
      pollTimer = window.setTimeout(refreshRun, POLL_INTERVAL_MS);
      return;
    }
    if (run.status === "starting") {
      approveRun.disabled = true;
      setWorking(firstPlannedAction(run), "Передаю план в VLA...");
      pollTimer = window.setTimeout(refreshRun, POLL_INTERVAL_MS);
      return;
    }
    if (run.status === "cancelling") {
      approveRun.disabled = true;
      setWorking(currentAction(run.events, run.task || taskText.textContent), "Останавливаю задачу...");
      pollTimer = window.setTimeout(refreshRun, POLL_INTERVAL_MS);
      return;
    }
    if (["failed", "rejected", "cancelled"].includes(run.status)) {
      stopPolling();
      isRunning = false;
      stopRequested = false;
      approveRun.disabled = true;
      stopRun.disabled = false;
      taskHint.textContent = terminalError(run);
      show(task);
      return;
    }

    approveRun.disabled = true;
    updateWorkingView(run);
    pollTimer = window.setTimeout(refreshRun, POLL_INTERVAL_MS);
  } catch (error) {
    setWorking(taskText.textContent, "Нет связи с оркестратором. Повторяю проверку...");
    pollTimer = window.setTimeout(refreshRun, POLL_INTERVAL_MS * 2);
  }
}

async function startRun(command) {
  const response = await fetch("/api/robot/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command, images: [] }),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.detail || `Ошибка ${response.status}`);
  if (!result.run_id) throw new Error("Оркестратор не вернул идентификатор запуска.");
  return result;
}

intro.addEventListener("click", openTask, { once: true });
intro.addEventListener(
  "keydown",
  (event) => {
    if (event.key === "Enter" || event.key === " ") openTask();
  },
  { once: true },
);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const command = input.value.trim();
  if (!command || isRunning) return;

  isRunning = true;
  currentRunId = null;
  stopRequested = false;
  approveRun.disabled = true;
  stopRun.disabled = false;
  taskHint.textContent = "Роборука поймёт обычную команду";
  setWorking(command, "Считываю сцену и строю план");

  try {
    const result = await startRun(command);
    currentRunId = result.run_id;
    if (stopRequested) {
      stopRequested = false;
      await cancelCurrentRun();
    }
    updateWorkingView({ task: command, status: result.run_status, events: [] });
    refreshRun();
  } catch (error) {
    isRunning = false;
    stopRequested = false;
    approveRun.disabled = true;
    stopRun.disabled = false;
    taskHint.textContent = error.message || "Не удалось запустить задачу.";
    show(task);
  }
});

again.addEventListener("click", () => {
  stopPolling();
  currentRunId = null;
  stopRequested = false;
  approveRun.disabled = true;
  stopRun.disabled = false;
  input.value = "";
  taskHint.textContent = "Роборука поймёт обычную команду";
  show(task);
  input.focus();
});

stopRun.addEventListener("click", cancelCurrentRun);
approveRun.addEventListener("click", approveCurrentRun);
