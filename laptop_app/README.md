# Laptop App

Это сайт и backend, которые запускаются на ноуте.

## 4090 Orchestrated Mode

When this folder is part of the `robot-camera-splitter` package on the 4090,
it is the only web UI. Start the complete stack from the package root:

```bash
chmod +x scripts/*.sh
./scripts/start_orchestrated_web.sh
```

The script starts the camera splitter, resident SmolVLA inference runner and
VLM orchestrator, then launches this site at `http://100.64.0.1:8000`.
Set these values in `laptop_app/.env` on the 4090:

```env
ORCHESTRATOR_ENABLED=1
ORCHESTRATOR_URL=http://127.0.0.1:8092
ORCHESTRATOR_TIMEOUT=330
WEB_HOST=0.0.0.0
WEB_PORT=8000
```

With orchestration enabled, a submitted command uses fresh splitter frames.
The VLM creates a contract, the runner receives one action at a time, and the
VLM verifies the scene before the next action. The site reports completion only
when the orchestrator returns `completed`.

After the VLM contract is ready, the working screen displays the first action.
`Подтвердить` gives one operator approval for the run; only then may the
orchestrator submit plan steps to the resident runner. `Стоп` cancels planning
or the active runner revision.

The resident runner is intentionally inference-only. It requires an existing
separate secured robot state and actuator deployment before camera observations
can become physical motion.

```powershell
cd "C:\Users\evangelino\Documents\для пака\laptop_app"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
python main.py
```

Открыть:

```text
http://127.0.0.1:8000
```

Что делает:

- `Проверить` - проверяет Qwen/VLM на `OPENAI_BASE_URL`.
- `Камеры` - проверяет splitter на `CAMERA_BASE_URL`.
- `➜` / Enter - берёт кадры с 4090, отправляет в VLM и показывает короткий план.
- `◎` - прикрепляет картинки вручную; если картинки прикреплены, отправка использует их вместо камер.

## SmolVLA

SmolVLA is a local LeRobot model, not an HTTP command endpoint. It must be launched on the 4090 with two `type: zmq` cameras. The exact camera replacement block is in [camera_4090/README.md](../camera_4090/README.md). Keep `VLA_ENABLED=0` here: this laptop app builds the visual plan; it does not pretend that an unloaded/nonexistent HTTP adapter is the VLA model.

## Action Contract

The planner returns `robot_action_contract.v1`. Every VLA action has a strict visual verification block: required visible objects, success predicates, failure predicates, and `on_uncertain: "stop"`. The predicate list is defined in [camera_4090/CONTRACT.md](../camera_4090/CONTRACT.md). Invalid JSON, an unapproved action, a direction word, or an unknown predicate is converted to a single `stop` step.

Allowed VLM action strings:

```text
move <object> to <object or destination>
grasp <object>
place to <object or destination>
put <object> into <object or destination>
put <object> on <object or destination>
stop
```

Direction words are rejected: `left`, `right`, `up`, `down`, `front`, `behind`, coordinates, pixels, angles.

## Model keepalive

Backend keeps the Qwen endpoint warm with tiny chat requests, so the model stays loaded by the OpenAI-compatible server.

Settings in `.env`:

```env
MODEL_KEEPALIVE_ENABLED=1
MODEL_KEEPALIVE_ON_STARTUP=1
MODEL_KEEPALIVE_INTERVAL=30
MODEL_KEEPALIVE_TIMEOUT=120
MODEL_KEEPALIVE_MAX_TOKENS=1
MODEL_KEEPALIVE_PROMPT=ok
```

Manual warmup from PowerShell:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/warmup
```

Warmup status:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/warmup
```
