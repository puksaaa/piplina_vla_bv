[CmdletBinding()]
param(
    [switch]$Install
)

$ErrorActionPreference = 'Stop'
$appDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $appDir '.venv\Scripts\python.exe'
$envFile = Join-Path $appDir '.env'
$envTemplate = Join-Path $appDir '.env.example'

Set-Location $appDir

if (-not (Test-Path $venvPython)) {
    Write-Host 'Creating virtual environment...'
    python -m venv (Join-Path $appDir '.venv')
}

if ($Install -or -not (Test-Path (Join-Path $appDir '.venv\.dependencies-installed'))) {
    Write-Host 'Installing dependencies...'
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r (Join-Path $appDir 'requirements.txt')
    New-Item -ItemType File -Path (Join-Path $appDir '.venv\.dependencies-installed') -Force | Out-Null
}

if (-not (Test-Path $envFile)) {
    Copy-Item -LiteralPath $envTemplate -Destination $envFile
    Write-Host 'Created laptop_app/.env from .env.example. Check camera and VLM addresses before using a robot.'
}

Write-Host 'Open http://127.0.0.1:8000'
& $venvPython (Join-Path $appDir 'main.py')
