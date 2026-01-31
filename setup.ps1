[CmdletBinding()]
param(
    [string]$OllamaHost = "0.0.0.0",
    [int]$OllamaPort = 11434,
    [string[]]$Models = @("gurubot/girl", "llama2-uncensored"),
    [string]$PythonExe = "python",
    [switch]$SkipBuildTools
)

function Assert-Administrator {
    $current = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($current)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Run this script from an elevated PowerShell session."
    }
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' not found in PATH."
    }
}

function Invoke-Winget {
    param(
        [string]$Id,
        [string]$Override = $null
    )
    $args = @(
        "install",
        "--id", $Id,
        "--exact",
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--silent"
    )
    if ($Override) {
        $args += @("--override", $Override)
    }
    Write-Host "[winget] Installing $Id..."
    $proc = Start-Process -FilePath "winget" -ArgumentList $args -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        throw "winget install for '$Id' failed with exit code $($proc.ExitCode)."
    }
}

function Ensure-Ollama {
    if (Get-Command "ollama" -ErrorAction SilentlyContinue) {
        Write-Host "[ollama] Already installed."
        return
    }
    Invoke-Winget -Id "Ollama.Ollama"
}

function Ensure-BuildTools {
    if ($SkipBuildTools) {
        Write-Host "[build-tools] Skipping install by request."
        return
    }
    $vsPath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer"
    if (Test-Path $vsPath) {
        Write-Host "[build-tools] Visual Studio infrastructure detected; continuing."
        return
    }
    $override = "--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools"
    Invoke-Winget -Id "Microsoft.VisualStudio.2022.BuildTools" -Override $override
}

function Configure-OllamaEnvironment {
    [Environment]::SetEnvironmentVariable("OLLAMA_HOST", $OllamaHost, "Machine")
    [Environment]::SetEnvironmentVariable("OLLAMA_PORT", $OllamaPort.ToString(), "Machine")
    Write-Host "[ollama] Persisted OLLAMA_HOST=$OllamaHost and OLLAMA_PORT=$OllamaPort."
}

function Ensure-FirewallRule {
    $ruleName = "Ollama HTTP $OllamaPort"
    if (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue) {
        Write-Host "[firewall] Rule '$ruleName' already exists."
        return
    }
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $OllamaPort | Out-Null
    Write-Host "[firewall] Opened TCP $OllamaPort for Ollama."
}

function Ensure-Venv {
    param([string]$Root)
    $venvPath = Join-Path $Root ".venv"
    if (-not (Test-Path $venvPath)) {
        Write-Host "[python] Creating virtual environment at $venvPath"
        & $PythonExe -m venv $venvPath
    } else {
        Write-Host "[python] Virtual environment already exists."
    }
    return $venvPath
}

function Install-PythonDependencies {
    param([string]$Root, [string]$VenvPath)
    $pip = Join-Path $VenvPath "Scripts/pip.exe"
    & $pip install --upgrade pip
    & $pip install -r (Join-Path $Root "requirements.txt")
}

function Pull-OllamaModels {
    param([string[]]$ModelList)
    foreach ($model in $ModelList) {
        Write-Host "[ollama] Pulling model $model"
        $proc = Start-Process -FilePath "ollama" -ArgumentList @("pull", $model) -Wait -PassThru
        if ($proc.ExitCode -ne 0) {
            throw "ollama pull for '$model' failed with exit code $($proc.ExitCode)."
        }
    }
}

function Ensure-OllamaServe {
    if (-not (Get-Process -Name "ollama" -ErrorAction SilentlyContinue)) {
        Write-Host "[ollama] Starting ollama serve in the background."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow
        Start-Sleep -Seconds 3
    } else {
        Write-Host "[ollama] Service already running."
    }
}

try {
    Assert-Administrator
    Require-Command -Name "winget"
    Require-Command -Name $PythonExe

    $root = Split-Path -Parent $MyInvocation.MyCommand.Path

    Ensure-Ollama
    Ensure-BuildTools
    Configure-OllamaEnvironment
    Ensure-FirewallRule
    $venv = Ensure-Venv -Root $root
    Install-PythonDependencies -Root $root -VenvPath $venv
    Ensure-OllamaServe
    Pull-OllamaModels -ModelList $Models

    Write-Host "\nSetup complete. Activate the virtual environment with:`n    .\\.venv\\Scripts\\Activate.ps1" -ForegroundColor Green
    Write-Host "Then start chatting via:`n    python src/ollama_voice.py --model $($Models[0])" -ForegroundColor Green
}
catch {
    Write-Error $_
    exit 1
}
