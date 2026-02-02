[CmdletBinding()]
param(
    [string]$OllamaHost = "0.0.0.0",
    [int]$OllamaPort = 11434,
    [string[]]$Models = @("deepseek-r1:7b"),
    [string]$PythonExe = "python",
    [string]$PythonWingetId = "Python.Python.3.11",
    [switch]$SkipBuildTools
)

function Assert-Administrator {
    $current = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($current)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Run this script from an elevated PowerShell session."
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
        if ($proc.ExitCode -eq -1978335230) {
            Write-Host "[winget] $Id exit code -1978335230; treating as already installed."
            return
        }

        # winget returns negative codes when the package exists or a reboot is pending; verify before failing
        $packagePresent = winget list --id $Id --exact 2>$null | Select-String $Id -Quiet
        if ($packagePresent) {
            Write-Host "[winget] $Id already present; continuing."
            return
        }
        throw "winget install for '$Id' failed with exit code $($proc.ExitCode)."
    }
}

function Ensure-Ollama {
    if (Get-Command "ollama" -ErrorAction SilentlyContinue) {
        Write-Host "[ollama] Already installed."
        return
    }

    $packagePresent = winget list --id "Ollama.Ollama" --exact 2>$null | Select-String "Ollama.Ollama" -Quiet
    if ($packagePresent) {
        Write-Host "[ollama] Package recorded in winget; skipping reinstall."
        return
    }
    Invoke-Winget -Id "Ollama.Ollama"
}

function Ensure-Python {
    param(
        [string]$Command,
        [string]$WingetId
    )

    $resolved = Get-Command $Command -ErrorAction SilentlyContinue
    if ($resolved -and $resolved.Source -notlike "*WindowsApps*") {
        return $resolved.Source
    }

    if ($resolved -and $resolved.Source -like "*WindowsApps*") {
        Write-Host "[python] Windows Store alias detected; installing real Python via winget."
    }
    elseif (-not $resolved) {
        Write-Host "[python] Python command missing; installing via winget."
    }

    Invoke-Winget -Id $WingetId

    $resolved = Get-Command $Command -ErrorAction SilentlyContinue
    if ($resolved -and $resolved.Source -notlike "*WindowsApps*") {
        return $resolved.Source
    }

    $installRoot = Join-Path ${env:LocalAppData} "Programs/Python"
    if (Test-Path $installRoot) {
        $candidates = Get-ChildItem -Path $installRoot -Directory | Sort-Object FullName -Descending
        foreach ($candidate in $candidates) {
            $exe = Join-Path $candidate.FullName "python.exe"
            if (Test-Path $exe) {
                return $exe
            }
        }
    }

    throw "Python installation failed; ensure Python 3.10+ is installed and accessible."
}

function Test-BuildToolsInstalled {
    $vsInstallerRoot = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer"
    $vswhere = Join-Path $vsInstallerRoot "vswhere.exe"
    if (Test-Path $vswhere) {
        $installationPath = & $vswhere -products Microsoft.VisualStudio.Product.BuildTools -latest -property installationPath 2>$null
        if ($LASTEXITCODE -eq 0 -and $installationPath) {
            return $true
        }
    }

    $defaultPath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/2022/BuildTools"
    if (Test-Path $defaultPath) {
        return $true
    }

    $wingetEntry = winget list --id "Microsoft.VisualStudio.2022.BuildTools" --exact 2>$null | Select-String "Microsoft.VisualStudio.2022.BuildTools" -Quiet
    if ($wingetEntry) {
        return $true
    }

    return $false
}

function Ensure-BuildTools {
    if ($SkipBuildTools) {
        Write-Host "[build-tools] Skipping install by request."
        return
    }
    if (Test-BuildToolsInstalled) {
        Write-Host "[build-tools] Build Tools already detected; skipping install."
        return
    }
    $override = "--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools"
    try {
        Invoke-Winget -Id "Microsoft.VisualStudio.2022.BuildTools" -Override $override
    }
    catch {
        if (Test-BuildToolsInstalled) {
            Write-Host "[build-tools] winget reported an error but Build Tools are present; continuing."
            return
        }
        throw
    }

    if (-not (Test-BuildToolsInstalled)) {
        Write-Warning "Visual Studio Build Tools installation could not be verified; continuing per user request."
    }
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
    param(
        [string]$Root,
        [string]$PythonPath
    )
    $venvPath = Join-Path $Root ".venv"
    if (-not (Test-Path $venvPath)) {
        Write-Host "[python] Creating virtual environment at $venvPath"
        & $PythonPath -m venv $venvPath
    } else {
        Write-Host "[python] Virtual environment already exists."
    }
    return $venvPath
}

function Install-PythonDependencies {
    param([string]$Root, [string]$VenvPath)
    $venvPython = Join-Path $VenvPath "Scripts/python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment python executable missing at $venvPython"
    }
    & $venvPython -m ensurepip --upgrade
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r (Join-Path $Root "requirements.txt")
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
    if (-not (Get-Command "winget" -ErrorAction SilentlyContinue)) {
        throw "Required command 'winget' not found in PATH."
    }

    $root = Split-Path -Parent $MyInvocation.MyCommand.Path
    $pythonPath = Ensure-Python -Command $PythonExe -WingetId $PythonWingetId

    Ensure-Ollama
    Ensure-BuildTools
    Configure-OllamaEnvironment
    Ensure-FirewallRule
    $venv = Ensure-Venv -Root $root -PythonPath $pythonPath
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
