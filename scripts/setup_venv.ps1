# Create a Windows Python virtual environment and install camera_create in editable mode.
param(
    [string]$PythonCommand = "py",
    [string]$EnvironmentPath = ".venv"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
& $PythonCommand -3.10 -m venv (Join-Path $ProjectRoot $EnvironmentPath)
$EnvironmentPython = Join-Path $ProjectRoot "$EnvironmentPath\Scripts\python.exe"
& $EnvironmentPython -m pip install --upgrade pip setuptools wheel
& $EnvironmentPython -m pip install -e "$ProjectRoot[dev]"
Write-Output "Environment ready. Activate with: $ProjectRoot\$EnvironmentPath\Scripts\Activate.ps1"

