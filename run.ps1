<#!
.SYNOPSIS
    Creates the local environment when needed, installs dependencies, and starts the demo.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run.ps1
#>

[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$checkpoint = Join-Path $projectRoot "models\best_bigru_attention_100.pth"

Set-Location $projectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.10-3.13 and make sure the 'python' command is available."
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating .venv..."
    python -m venv .venv
}

if (-not (Test-Path $checkpoint)) {
    throw "Model checkpoint was not found: $checkpoint"
}

Write-Host "Installing/checking dependencies..."
& $venvPython -m pip install --disable-pip-version-check -r requirements.txt

Write-Host "Starting Speaker Identification at http://127.0.0.1:$Port"
Write-Host "Press Ctrl+C to stop the server."
& $venvPython -m uvicorn app:app --host 127.0.0.1 --port $Port
