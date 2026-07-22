# Plan-Only Web Chat

The chat takes uploaded images or the latest local camera frame, sends them to Gemma, and returns a strict robot action contract. `VLA_ENABLED=0` means it never sends anything to an actuator.

Run it from the parent `camera_4090` directory:

```bash
./scripts/start_web_chat.sh
```

Open `http://100.64.0.1:8000`. Detailed request logs are in `web_app/logs/planner.log` and in the terminal.
