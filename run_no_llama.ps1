param(
    [switch]$PushToTalk,
    [string]$Model = "deepseek-r1:7b",
    [string]$OllamaHost = "127.0.0.1",
    [int]$OllamaPort = 11434
)

if (-not $PSBoundParameters.ContainsKey('PushToTalk')) {
    $PushToTalk = $true
}

# Ensures setup runs, then launches the Ollama voice client (optionally push-to-talk) with the project venv.
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$setupScript = Join-Path $repoRoot "setup.ps1"
$venvActivate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $setupScript)) {
    throw "Setup script not found at $setupScript"
}

Write-Host "Running setup script..." -ForegroundColor Cyan
& $setupScript

if (-not (Test-Path $venvActivate)) {
    throw "Virtual environment activation script not found at $venvActivate. Re-run setup manually if it failed to create."
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& $venvActivate

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment interpreter not found at $venvPython. Ensure the venv is created successfully."
}

# Pin the Ollama endpoint for this session to avoid relying on machine-wide state.
$env:OLLAMA_HOST = $OllamaHost
$env:OLLAMA_PORT = $OllamaPort.ToString()

$arguments = @("src/ollama_voice.py", "--model", $Model)
if ($PushToTalk) {
    $arguments = @("src/ollama_voice.py", "--push-to-talk", "--model", $Model)
}

Write-Host "Starting Ollama voice client..." -ForegroundColor Cyan
& $venvPython @arguments
