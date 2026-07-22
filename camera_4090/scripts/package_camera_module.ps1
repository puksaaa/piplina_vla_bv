$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$dist = Join-Path $root "dist"
$stage = Join-Path $dist "robot-camera-splitter"
$zip = Join-Path $dist "robot-camera-splitter.zip"

if (Test-Path $stage) {
  Remove-Item -Recurse -Force $stage
}
New-Item -ItemType Directory -Force -Path $stage | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "scripts") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "systemd") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "tests") | Out-Null

Copy-Item (Join-Path $root "camera_splitter.py") $stage
Copy-Item (Join-Path $root "verify_camera_splitter.py") $stage
Copy-Item (Join-Path $root "vlm_terminal.py") $stage
Copy-Item (Join-Path $root "supervisor.py") $stage
Copy-Item (Join-Path $root "smolvla_runner.py") $stage
Copy-Item (Join-Path $root "orchestrator.py") $stage
Copy-Item (Join-Path $root "verify_smolvla_runner.py") $stage
Copy-Item (Join-Path $root "verify_gemma.py") $stage
Copy-Item (Join-Path $root "requirements.txt") $stage
Copy-Item (Join-Path $root ".env.camera.example") $stage
Copy-Item (Join-Path $root ".env.vlm.example") $stage
Copy-Item (Join-Path $root ".env.smolvla.example") $stage
Copy-Item (Join-Path $root ".env.orchestrator.example") $stage
Copy-Item (Join-Path $root ".env.lerobot_rollout.example") $stage
Copy-Item (Join-Path $root "README.md") $stage
Copy-Item (Join-Path $root "RUNNER_API.md") $stage
Copy-Item (Join-Path $root "CONTRACT.md") $stage
Copy-Item (Join-Path (Split-Path $root -Parent) "laptop_app") $stage -Recurse
Get-ChildItem -Path $stage -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
$laptopVenv = Join-Path $stage "laptop_app\.venv"
if (Test-Path $laptopVenv) {
  Remove-Item -Recurse -Force $laptopVenv
}
$laptopEnv = Join-Path $stage "laptop_app\.env"
if (Test-Path $laptopEnv) {
  Remove-Item -Force $laptopEnv
}
$appLogs = @(
  (Join-Path $stage "laptop_app\logs")
)
foreach ($appLog in $appLogs) {
  if (Test-Path $appLog) {
    Remove-Item -Recurse -Force $appLog
  }
}
Copy-Item (Join-Path $root "scripts\install_camera_splitter.sh") (Join-Path $stage "scripts")
Copy-Item (Join-Path $root "scripts\start_camera_splitter.sh") (Join-Path $stage "scripts")
Copy-Item (Join-Path $root "scripts\start_vlm_terminal.sh") (Join-Path $stage "scripts")
Copy-Item (Join-Path $root "scripts\start_supervisor.sh") (Join-Path $stage "scripts")
Copy-Item (Join-Path $root "scripts\start_smolvla_runner.sh") (Join-Path $stage "scripts")
Copy-Item (Join-Path $root "scripts\start_orchestrator.sh") (Join-Path $stage "scripts")
Copy-Item (Join-Path $root "scripts\start_orchestrated_web.sh") (Join-Path $stage "scripts")
Copy-Item (Join-Path $root "scripts\start_pipeline.sh") (Join-Path $stage "scripts")
Copy-Item (Join-Path $root "scripts\start_lerobot_rollout_zmq.sh") (Join-Path $stage "scripts")
Copy-Item (Join-Path $root "systemd\camera-splitter@.service") (Join-Path $stage "systemd")
Copy-Item (Join-Path $root "tests\test_smolvla_supervisor.py") (Join-Path $stage "tests")

$readme = @(
  '# Robot Camera Splitter',
  '',
  'The complete deployment guide is in README.md.',
  '',
  '```bash',
  'chmod +x scripts/*.sh',
  './scripts/install_camera_splitter.sh',
  '```',
  'Full web pipeline:',
  '',
  '```bash',
  './scripts/start_orchestrated_web.sh',
  '```',
  '',
  'Manual runner verification:',
  '',
  '```bash',
  'python verify_smolvla_runner.py',
  '```'
)

$readme | Set-Content -Encoding UTF8 (Join-Path $stage "README_CAMERA_MODULE.md")

if (Test-Path $zip) {
  Remove-Item -Force $zip
}
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip

Write-Host "Created $zip"
