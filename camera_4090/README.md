# Robot Camera Splitter для 4090

Этот архив нужен для компьютера с 4090. Он поднимает разветвитель USB-камер и веб-страницу, где можно написать задачу роботу. VLM получает свежий кадр со сцены или загруженную картинку и возвращает короткий контракт:

- `plan[].action`: простые команды для VLA, например `move red cube to white plate`;
- `verification.success`: как понять, что шаг выполнен;
- `verification.failure`: как понять, что шаг явно провален, например объект не оказался на цели или упал;
- `uncertain`: если объект/цель не видно или кадр сомнительный, это не `failed`, а неопределенность.

USB-камеры открывает только `camera_splitter.py`. SmolVLA получает кадры через ZMQ-разветвитель, а VLM/Gemma получает HTTP-снимки. Резидентный runner держит policy в CUDA. Актуация опциональна (`SMOLVLA_ACTUATION_ENABLED`): по умолчанию runner только считает action chunks, а при включении сам открывает serial-порт SO-101, читает состояние суставов и отправляет действия на моторы (см. раздел 10).

## 1. Что перенести на 4090

Перенеси файл:

```text
camera_4090/dist/robot-camera-splitter.zip
```

На 4090 распакуй его в удобную папку:

```bash
cd ~/Documents
mkdir -p robot-camera-splitter
unzip ~/Downloads/robot-camera-splitter.zip -d robot-camera-splitter
cd robot-camera-splitter
chmod +x scripts/*.sh
```

Если архив лежит в другой папке, замени путь `~/Downloads/robot-camera-splitter.zip` на свой.

## 2. Поставить окружение

На Ubuntu/Debian сначала поставь базовые пакеты:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip curl v4l-utils unzip
```

Потом создай `.venv` и поставь зависимости:

```bash
./scripts/install_camera_splitter.sh
```

Скрипт сам сделает:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.camera.example .env.camera
```

## 3. Найти индексы USB-камер

Активируй окружение и просканируй камеры:

```bash
source .venv/bin/activate
python camera_splitter.py --list-cameras
```

Ищи строки вида:

```text
1: ok (640x480)
2: ok (640x480)
```

Если у тебя `1` и `2` показывают `ok`, значит в `.env.camera` надо поставить:

```env
CAMERA_FRONT_DEVICE=1
CAMERA_WRIST_DEVICE=2
```

Если `ok` у других индексов, ставь их. Например, если `0` и `2`:

```env
CAMERA_FRONT_DEVICE=0
CAMERA_WRIST_DEVICE=2
```

Открыть конфиг:

```bash
nano .env.camera
```

Минимальный рабочий `.env.camera`:

```env
CAMERA_FRONT_DEVICE=1
CAMERA_WRIST_DEVICE=2

CAMERA_FRONT_NAME=camera1
CAMERA_WRIST_NAME=camera2

CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=15
CAMERA_JPEG_QUALITY=82

CAMERA_HTTP_HOST=0.0.0.0
CAMERA_HTTP_PORT=8090

CAMERA_ENABLE_ZMQ=1
CAMERA_ZMQ_BIND=tcp://0.0.0.0:5555
CAMERA_ZMQ_FPS=15
```

`CAMERA_FRONT_NAME` и `CAMERA_WRIST_NAME` должны совпадать с именами камер, с которыми обучали SmolVLA. Если обучение было на `camera1` и `camera2`, оставь так.

Если доступна только одна камера, сайт и VLM смогут работать с одной картинкой: укажи рабочий индекс хотя бы в одном поле. Второй reader будет помечен как missing. Для SmolVLA, обученной на двух камерах, одной камеры обычно недостаточно.

## 4. Запустить только разветвитель камер

В отдельном терминале:

```bash
cd ~/Documents/robot-camera-splitter
./scripts/start_camera_splitter.sh
```

Проверка на самой 4090:

```bash
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/snapshot/all | head -c 500
```

Открыть preview с ноутбука через Tailscale:

```text
http://100.64.0.1:8090/preview
```

Если preview показывает обе картинки, splitter работает.

## 5. Как теперь отдавать камеры в VLA/LeRobot

Раньше VLA открывала USB напрямую, например через `opencv` и индекс камеры. Теперь так делать нельзя, иначе будет конфликт: USB-камеру должен открыть только splitter.

Новая логика:

```text
USB camera 1 -> camera_splitter -> ZMQ camera1 -> LeRobot / SmolVLA
USB camera 2 -> camera_splitter -> ZMQ camera2 -> LeRobot / SmolVLA
                                -> HTTP snapshots -> VLM / Gemma
```

То есть ребятам нужно заменить USB/OpenCV камеры в конфиге VLA на ZMQCamera:

```text
camera1:
  type: zmq
  host: 127.0.0.1
  port: 5555
  name: camera1
  width: 640
  height: 480

camera2:
  type: zmq
  host: 127.0.0.1
  port: 5555
  name: camera2
  width: 640
  height: 480
```

Точный синтаксис зависит от их LeRobot-конфига, но смысл такой: вместо `/dev/video0`, `/dev/video2` или `opencv` указываются ZMQ host/port/name. Имена `camera1` и `camera2` должны совпадать с `.env.camera`.

### Замена твоего текущего `lerobot-rollout`

Было:

```bash
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower_arm \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --policy.path=outputs/train/my_smolvla/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --task="Pick up the red cube and place it in the box" \
  --duration=60 \
  --display_data=true
```

Должно стать через splitter/ZMQ:

```bash
lerobot-rollout \
  --strategy.type=base \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower_arm \
  --robot.cameras="{front: {type: zmq, server_address: '127.0.0.1', port: 5555, camera_name: 'camera1', width: 640, height: 480, fps: 15}}" \
  --policy.path=outputs/train/my_smolvla/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --task="Pick up the red cube and place it in the box" \
  --duration=60 \
  --display_data=true
```

Если policy обучалась на двух камерах:

```bash
lerobot-rollout \
  --strategy.type=base \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower_arm \
  --robot.cameras="{front: {type: zmq, server_address: '127.0.0.1', port: 5555, camera_name: 'camera1', width: 640, height: 480, fps: 15}, wrist: {type: zmq, server_address: '127.0.0.1', port: 5555, camera_name: 'camera2', width: 640, height: 480, fps: 15}}" \
  --policy.path=outputs/train/my_smolvla/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --task="Pick up the red cube and place it in the box" \
  --duration=60 \
  --display_data=true
```

Важно:

- `front` / `wrist` слева в `--robot.cameras` должны совпадать с camera keys, которые видела policy при обучении;
- `camera_name: 'camera1'` / `camera_name: 'camera2'` должны совпадать с `CAMERA_FRONT_NAME` / `CAMERA_WRIST_NAME` в `.env.camera`;
- `server_address: '127.0.0.1'` норм, если splitter и VLA запущены на той же 4090;
- `fps` в LeRobot лучше держать таким же, как `CAMERA_ZMQ_FPS` в `.env.camera`.

Я добавил короткий запуск через конфиг:

```bash
cp .env.lerobot_rollout.example .env.lerobot_rollout
nano .env.lerobot_rollout
./scripts/start_lerobot_rollout_zmq.sh "Pick up the red cube and place it in the box"
```

В `.env.lerobot_rollout` для одной камеры оставь:

```env
LEROBOT_CAMERA_FRONT_KEY=front
LEROBOT_CAMERA_FRONT_NAME=camera1
LEROBOT_CAMERA_WRIST_ENABLED=0
```

Для двух камер:

```env
LEROBOT_CAMERA_FRONT_KEY=front
LEROBOT_CAMERA_FRONT_NAME=camera1
LEROBOT_CAMERA_WRIST_ENABLED=1
LEROBOT_CAMERA_WRIST_KEY=wrist
LEROBOT_CAMERA_WRIST_NAME=camera2
```

Скрипт `start_lerobot_rollout_zmq.sh` сам поднимет splitter, дождется кадра и только потом запустит `lerobot-rollout`.

## 6. Настроить полный веб-пайплайн

Создай `.env` для сайта:

```bash
cp laptop_app/.env.example laptop_app/.env
nano laptop_app/.env
```

Проверь главные строки:

```env
OPENAI_BASE_URL=http://100.64.0.4:1234/v1
OPENAI_API_KEY=dummy
OPENAI_MODEL=google/gemma-4-31b
ROBOT_VLM_MODEL=google/gemma-4-31b
OPENAI_TIMEOUT=300

CAMERA_BASE_URL=http://127.0.0.1:8090
CAMERA_TIMEOUT=10

ORCHESTRATOR_ENABLED=1
ORCHESTRATOR_URL=http://127.0.0.1:8092
ORCHESTRATOR_TIMEOUT=330
WEB_HOST=0.0.0.0
WEB_PORT=8000
```

`OPENAI_BASE_URL` должен смотреть на ручку модели через Tailscale. В этом проекте это:

```text
http://100.64.0.4:1234/v1
```

Создай также конфиги резидентного runner и самого оркестратора:

```bash
cp .env.smolvla.example .env.smolvla
cp .env.orchestrator.example .env.orchestrator
nano .env.smolvla
```

В `.env.smolvla` обязательно укажи путь к fine-tuned checkpoint:

```env
SMOLVLA_POLICY_PATH=/absolute/path/to/fine_tuned_checkpoint/pretrained_model
SMOLVLA_DEVICE=cuda
SMOLVLA_CAMERA_NAMES=camera1,camera2
SMOLVLA_CAMERA_WIDTH=640
SMOLVLA_CAMERA_HEIGHT=480
```

Нужен именно каталог, внутри которого лежит `config.json`, обычно это
`.../checkpoints/last/pretrained_model`, а не корень `outputs/train/...`.
Найти его на 4090 можно так:

```bash
find ~/Documents -type f -path "*/pretrained_model/config.json" -print
```

Если команда вернула, например,
`/home/lab/Documents/lerobot/outputs/train/my_smolvla/checkpoints/last/pretrained_model/config.json`,
запиши в `.env.smolvla` строку без последнего `/config.json`:

```env
SMOLVLA_POLICY_PATH=/home/lab/Documents/lerobot/outputs/train/my_smolvla/checkpoints/last/pretrained_model
```

Если LeRobot находится в отдельном окружении, задай и его Python:

```env
SMOLVLA_PYTHON=/home/lab/miniconda3/envs/lerobot/bin/python
```

## 7. Запустить сайт, VLM и resident SmolVLA

Одна команда:

```bash
cd ~/Documents/robot-camera-splitter
./scripts/start_orchestrated_web.sh
```

Скрипт:

1. проверит все четыре конфигурации и `laptop_app/.env`;
2. запустит splitter и дождётся свежих кадров;
3. загрузит SmolVLA в CUDA один раз и проверит совместимость имён/размера камер с metadata checkpoint;
4. разогреет VLM/Gemma и поднимет оркестратор;
5. поднимет единственный сайт `laptop_app` на `0.0.0.0:8000`.

Открыть сайт с ноутбука:

```text
http://100.64.0.1:8000
```

На сайте напиши команду, например `перемести все кубики в тарелку`. Он возьмёт свежие кадры, создаст VLM-contract, будет передавать runner только один action за раз, ждать выделенное на него время, затем визуально проверит сцену. При `completed` перейдёт к следующему действию; при ошибке, неуверенности или невыполнении отменит текущую ревизию и построит новый оставшийся план.

После построения плана сайт показывает первый VLA-шаг и две операторские кнопки:
`Подтвердить` и `Стоп`. До подтверждения runner не получает ни одного шага.
Одно подтверждение разрешает оркестратору последовательно передавать весь план,
визуально проверять действия и перепланировать оставшуюся работу.

`Стоп` отменяет весь текущий run: если VLM ещё строит план, ни один VLA-шаг не
будет запущен; если шаг уже активен, supervisor отменит его текущую ревизию.
Это программная отмена задачи, а не физическая аварийная остановка приводов.

## 8. Где смотреть логи

Если сайт долго думает, это нормально для Gemma/VLM. Смотри живые логи:

```bash
tail -f laptop_app/logs/planner.log
```

Логи сервисов полного запуска:

```bash
tail -f logs/camera-splitter.log
tail -f logs/smolvla-runner.log
tail -f logs/orchestrator.log
```

В логах сайта важные события:

```text
event=plan_started
event=images_ready
event=vlm_request_started
event=vlm_raw_response
event=vlm_request_finished
```

Если есть `vlm_raw_response`, значит модель ответила. Если сайт показал `stop`, смотри `failure_reason`.

## 9. Проверка без сайта

Проверить splitter:

```bash
source .venv/bin/activate
python verify_camera_splitter.py --base-url http://127.0.0.1:8090 --zmq-address tcp://127.0.0.1:5555 --cameras camera1,camera2
```

Проверить ручку Gemma:

```bash
source .venv/bin/activate
python verify_gemma.py
```

Проверить терминальный VLM-план по камерам:

```bash
source .venv/bin/activate
python vlm_terminal.py "перемести красный кубик в белую тарелку"
```

## 10. Resident SmolVLA runner

Почему это отдельный слой: `lerobot-rollout --task="..."` получает task на старте процесса. Это отлично для проверки, что VLA, рука и ZMQ-камеры работают. Но для полноценного пайплайна `VLM построила contract -> VLA выполняет step 1 -> VLM проверяет -> VLA выполняет step 2` нельзя удобно менять `--task` внутри уже запущенного `lerobot-rollout`. Поэтому для production-пайплайна нужен resident runner: модель загружается в CUDA один раз, а действия приходят как ревизии.

Создать конфиг:

```bash
cp .env.smolvla.example .env.smolvla
nano .env.smolvla
```

Главные поля:

```env
SMOLVLA_POLICY_PATH=/absolute/path/to/fine_tuned_checkpoint
SMOLVLA_PYTHON=/absolute/path/to/lerobot/env/bin/python
SMOLVLA_DEVICE=cuda

SMOLVLA_CAMERA_HOST=127.0.0.1
SMOLVLA_CAMERA_PORT=5555
SMOLVLA_CAMERA_NAMES=camera1,camera2
SMOLVLA_CAMERA_WIDTH=640
SMOLVLA_CAMERA_HEIGHT=480
```

Проверить runner:

```bash
./scripts/start_smolvla_runner.sh
```

Health:

```bash
curl http://127.0.0.1:8091/health
```

Runner держит SmolVLA в CUDA-памяти один раз и принимает action/revision от supervisor.

### Актуация (реальное движение руки)

По умолчанию актуация выключена: runner только считает action chunks и ждёт
состояние робота через `POST /v1/state`. Чтобы он **реально двигал SO-101**,
включи актуацию в `.env.smolvla`:

```env
SMOLVLA_ROBOT_TYPE=so101_follower
SMOLVLA_ROBOT_PORT=/dev/ttyACM0
SMOLVLA_ROBOT_ID=robot_arm

SMOLVLA_ACTUATION_ENABLED=1
SMOLVLA_MAX_RELATIVE_TARGET=5     # маленький кламп на первых запусках
SMOLVLA_ROBOT_USE_DEGREES=1
```

Тогда единственный поток управления runner сам открывает serial-порт (только он,
камеры при этом остаются за splitter), каждый тик читает реальные позиции
суставов как `observation.state`, гоняет policy и отправляет действие на моторы
через `robot.send_action`. Камеры по-прежнему из ZMQ, чекпоинт грузится как
раньше.

Безопасный порядок первого запуска на железе:

1. Сначала **dry-run без движения** — подключиться и убедиться, что порядок
   суставов и чтение состояния верные:

   ```bash
   source .venv/bin/activate
   python verify_actuation.py --port /dev/ttyACM0
   ```

   Проверь в выводе `motor_pos_names`: порядок должен совпадать с порядком, на
   котором обучалась policy (`observation.state` / `action`).

2. Проверить сам путь отправки команды без видимого движения (рука держит
   текущую позицию):

   ```bash
   python verify_actuation.py --port /dev/ttyACM0 --send-hold
   ```

3. Крошечный аккуратный тест движения одного сустава и возврат:

   ```bash
   python verify_actuation.py --port /dev/ttyACM0 --nudge 3 --max-relative-target 5
   ```

4. Только после этого включай `SMOLVLA_ACTUATION_ENABLED=1` и запускай полный
   пайплайн. Держи руку под наблюдением и рядом с `Стоп`.

Промежуточный режим для отладки: `SMOLVLA_ROBOT_CONNECT=1` при
`SMOLVLA_ACTUATION_ENABLED=0` — runner подключается к руке и читает состояние,
логирует, какие действия **были бы** отправлены (`actuated=false`), но не двигает
моторы.

> Внимание: этот код ещё не проверялся на реальном железе. Порядок
> суставов, единицы (градусы/сырые) и соответствие размерностей
> `observation.state`/`action` обязательно провалидируй шагами выше перед первым
> движением.

Полный pipeline без веб-интерфейса, когда runner уже настроен:

```bash
./scripts/start_pipeline.sh "перемести красный кубик в белую тарелку" --log runs/session.jsonl
```

## 11. Частые проблемы

`python3 -m venv` не работает:

```bash
sudo apt install -y python3-venv
./scripts/install_camera_splitter.sh
```

`snapshot/all` возвращает `Frames are not ready`:

```bash
python camera_splitter.py --list-cameras
nano .env.camera
```

Проверь, что в `.env.camera` стоят индексы со статусом `ok`.

Камера занята:

```bash
sudo fuser -v /dev/video0 /dev/video1 /dev/video2
```

Останови процесс, который держит USB-камеру. VLA не должна открывать USB напрямую, пока работает splitter.

Порт уже занят:

```bash
sudo lsof -i :8090
sudo lsof -i :8000
```

Нет связи с моделью:

```bash
curl http://100.64.0.4:1234/v1/models
```

Если curl не отвечает, проблема не в сайте: проверь Tailscale и что model server на `100.64.0.4:1234` поднят.

## 12. Быстрый чек-лист

```bash
cd ~/Documents/robot-camera-splitter
chmod +x scripts/*.sh
./scripts/install_camera_splitter.sh
source .venv/bin/activate
python camera_splitter.py --list-cameras
nano .env.camera
cp laptop_app/.env.example laptop_app/.env
nano laptop_app/.env
cp .env.smolvla.example .env.smolvla
nano .env.smolvla
cp .env.orchestrator.example .env.orchestrator
./scripts/start_orchestrated_web.sh
```

Smoke-test VLA через ZMQ вместо USB/OpenCV:

```bash
cp .env.lerobot_rollout.example .env.lerobot_rollout
nano .env.lerobot_rollout
./scripts/start_lerobot_rollout_zmq.sh "Pick up the red cube and place it in the box"
```

Открыть с ноутбука:

```text
http://100.64.0.1:8000
```

Preview камер:

```text
http://100.64.0.1:8090/preview
```
