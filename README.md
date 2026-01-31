# Ollama Voice Chat

Interactive Ollama client for Windows that speaks responses through Coqui TTS. The project includes an automated setup script that prepares Ollama for remote access, installs the required voice packages, and downloads the DeepSeek-R1 7B model.

## Features

- Installs and configures Ollama with firewall access for LAN clients.
- Downloads DeepSeek-R1 7B by default.
- Sets up a Python virtual environment with Coqui TTS dependencies.
- Provides a CLI chat loop that plays each reply aloud.

## Requirements

- Windows 11 with administrator rights.
- winget available in PATH.
- Python 3.10 or later.
- At least 15 GB free disk space (Coqui voice model plus Ollama models).
- Speakers or headphones on the host machine.

## Quick Start

1. Open an elevated PowerShell in the project directory.
2. Run the automated setup:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   ./setup.ps1
   ```

3. Activate the virtual environment and start the voice chat:

   ```powershell
   ./.venv/Scripts/Activate.ps1
   python src/ollama_voice.py --model deepseek-r1:7b
   ```

4. For remote clients, pass the machine's LAN IP to the `--host` flag when launching the chat script.

## Automated Setup Script

`setup.ps1` performs the following steps:

1. Installs Ollama via winget if it is missing (`winget install --id Ollama.Ollama`).
2. Installs Visual Studio Build Tools with the VC toolchain (skip with `-SkipBuildTools`).
3. Persists `OLLAMA_HOST=0.0.0.0` and `OLLAMA_PORT=11434` for remote access.
4. Opens Windows Firewall for TCP 11434.
5. Creates a `.venv` virtual environment and installs packages from `requirements.txt`.
6. Starts `ollama serve` in the background and pulls the models listed in the script parameters.

The script defaults to downloading `deepseek-r1:7b`. Override the list with `-Models` if you need different models.

## Manual Installation

If you prefer to run the steps yourself, execute the following commands in an elevated PowerShell session:

1. Install Ollama:

   ```powershell
   winget install --id Ollama.Ollama --exact --accept-package-agreements --accept-source-agreements --silent
   ```

2. Install VC Build Tools (required by Coqui TTS):

   ```powershell
   winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --accept-package-agreements --accept-source-agreements --silent --override "--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools"
   ```

3. Set Ollama environment variables and open the firewall:

   ```powershell
   [Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "Machine")
   [Environment]::SetEnvironmentVariable("OLLAMA_PORT", "11434", "Machine")
   New-NetFirewallRule -DisplayName "Ollama HTTP 11434" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434
   ```

4. Start the Ollama service and download models:

   ```powershell
   ollama serve
   # In a new window
   ollama pull deepseek-r1:7b
   ```

5. Create a virtual environment and install Python dependencies:

   ```powershell
   python -m venv .venv
   ./.venv/Scripts/Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

After setup, launch the chat loop (activate the virtual environment first):

```powershell
python src/ollama_voice.py --host 127.0.0.1 --model deepseek-r1:7b
```

Key flags:

- `--host`: Ollama server host (use the machine's LAN IP for remote clients).
- `--port`: Ollama port (defaults to 11434).
- `--model`: Ollama model to query.
- `--tts-model`: Coqui TTS voice (defaults to `tts_models/en/jenny/jenny`).
- `--audio-device`: Optional sounddevice target (index or name).

## Verifying Remote Access

1. On the host machine, confirm the API is reachable:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:11434/api/tags
   ```

2. From a remote client on the same network, replace `127.0.0.1` with the host's IP. You should receive the same JSON listing the installed models.

## Troubleshooting

- If the script reports missing `winget`, install the latest App Installer from the Microsoft Store.
- `WinError 10049` indicates you are pointing the chat script at `0.0.0.0`; use `127.0.0.1` locally or the actual LAN IP remotely.
- If Coqui TTS installation fails, rerun `setup.ps1` without `-SkipBuildTools` to ensure the VC toolchain is present.
- For audio issues, list devices with:

  ```powershell
  python -c "import sounddevice as sd; print(sd.query_devices())"
  ```

  Then pass the desired index to `--audio-device`.
