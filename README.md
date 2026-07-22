# Robot VLM Planner Project

## Current Pipeline

Laptop backend now uses this flow:

```text
browser command
  -> POST /api/robot/run
  -> fetch latest camera frames from http://100.64.0.1:8090/snapshot/all
  -> send frames + command to VLM at http://100.64.0.4:1234/v1
  -> validate short object-based actions
  -> show the plan; the local SmolVLA runner consumes the ZMQ camera stream separately
```

Allowed VLM/VLA command strings:

```text
move <object> to <object or destination>
grasp <object>
place to <object or destination>
put <object> into <object or destination>
put <object> on <object or destination>
stop
```

Forbidden: left, right, up, down, front, behind, coordinates, pixels, angles.

SmolVLA is a local LeRobot model, not an HTTP robot-hand endpoint. It runs on the 4090 and receives frames through `ZMQCamera`. The resident runner (`camera_4090/smolvla_runner.py`) now closes the loop: with `SMOLVLA_ACTUATION_ENABLED=1` it opens the SO-101 serial port, reads joint state, and sends policy actions to the motors. See `camera_4090/README.md` section 10 for the opt-in flags and the safe first-run procedure on hardware.

Проект разделён по машинам:

```text
laptop_app/   - сайт и backend на твоём ноуте
camera_4090/  - camera splitter для Linux ПК с 4090 и двумя USB-камерами
DGX/ручка     - файлов не нужно, используется только OpenAI-compatible endpoint
```

Режимы: базовый режим ноутбука строит только план (`vla_called: false`).
Оркестрированный режим на 4090 реально вызывает SmolVLA (инференс), а при
`SMOLVLA_ACTUATION_ENABLED=1` ещё и физически двигает руку через resident runner.

## Схема

```text
ноут / laptop_app
  UI: команда для робота
    ↓
  FastAPI /api/robot/plan
    ↓
  GET http://100.64.0.1:8090/snapshot/all
    ↓
  кадры camera1 + camera2 с 4090
    ↓
  Qwen3.6 35B VLM на http://100.64.0.4:1234/v1
    ↓
  scene + deterministic primitive plan
```

```text
4090 / camera_4090
  USB cam 1 + USB cam 2
    ↓
  camera_splitter.py
    ├─ HTTP snapshots для VLM: http://100.64.0.1:8090
    └─ ZMQ stream для запущенной LeRobot SmolVLA: tcp://127.0.0.1:5555
```

## Что куда класть

### Ноут

Нужна папка:

```text
laptop_app/
```

Внутри:

```text
main.py
requirements.txt
.env.example
.env
static/
```

Запуск на ноуте:

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

В интерфейсе:

- `Проверить` - проверяет Qwen/VLM endpoint.
- `Камеры` - проверяет camera splitter на 4090.
- `◎` - прикрепляет картинки вручную; если они выбраны, отправка использует их вместо camera splitter.
- `➜` / Enter - берёт команду, кадры с камер или прикреплённые картинки, отправляет в VLM и показывает только primitive plan.

### 4090 Linux ПК

Нужна папка:

```text
camera_4090/
```

Внутри:

```text
camera_splitter.py
verify_camera_splitter.py
requirements.txt
.env.camera.example
scripts/
systemd/
```

Можно перенести всю папку `camera_4090`, либо собрать zip с ноутбука:

```powershell
cd "C:\Users\evangelino\Documents\для пака"
powershell -ExecutionPolicy Bypass -File .\camera_4090\scripts\package_camera_module.ps1
```

Zip появится здесь:

```text
camera_4090\dist\robot-camera-splitter.zip
```

Запуск на 4090:

```bash
cd /path/to/camera_4090
chmod +x scripts/*.sh
./scripts/install_camera_splitter.sh
source .venv/bin/activate
python camera_splitter.py --list-cameras
nano .env.camera
./scripts/start_camera_splitter.sh
```

В `.env.camera` обычно нужно поправить индексы:

```env
CAMERA_FRONT_DEVICE=0
CAMERA_WRIST_DEVICE=1
```

Проверка на 4090:

```bash
python verify_camera_splitter.py --base-url http://127.0.0.1:8090 --zmq-address tcp://127.0.0.1:5555
```

Предпросмотр камер:

```text
http://127.0.0.1:8090/preview
```

Проверка с ноута через Tailscale:

```powershell
cd "C:\Users\evangelino\Documents\для пака\camera_4090"
python verify_camera_splitter.py --base-url http://100.64.0.1:8090 --zmq-address tcp://100.64.0.1:5555
```

### DGX / ручка / Qwen

Файлы туда класть не нужно.

Ноут просто обращается к endpoint из `laptop_app\.env`:

```env
OPENAI_BASE_URL=http://100.64.0.4:1234/v1
OPENAI_API_KEY=dummy
OPENAI_MODEL=google/gemma-4-31b
ROBOT_VLM_MODEL=google/gemma-4-31b
```

## API

Ноут:

```text
GET  /health
GET  /api/models
GET  /api/camera/health
POST /api/chat
POST /api/robot/plan
POST /api/robot/run
```

4090:

```text
GET /health
GET /snapshot/all
GET /snapshot/front
GET /snapshot/wrist
GET /frame/front.jpg
GET /frame/wrist.jpg
GET /preview
```

Ответ `/api/robot/plan` всегда подчёркивает, что VLA не вызывается:

```json
{
  "mode": "plan_only",
  "vla_called": false,
  "scene": {},
  "plan": [],
  "uncertainties": [],
  "camera": {}
}
```
