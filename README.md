# HandyAidGUI

A desktop GUI for [Handy](https://github.com/cjpais/Handy)'s CLI backend — run, configure, and monitor transcription jobs without touching the command line.

> **Note:** This project is AI-coded. It works for my own use, but I'd recommend forking it if you plan to rely on it long-term rather than depending on this repo directly. Issues and PRs are welcome.

## What it does

- Wraps the Handy CLI's `transcribe-cli` binary with a GUI you point at audio/video input
- **Multi-GPU support** for systems with more than one Vulkan-capable GPU:
  - Detects every physical GPU adapter and shows a live VRAM meter and eligibility status (whether a GPU has enough free VRAM for the loaded model) for each
  - **Auto mode** splits a job across GPUs weighted by free VRAM
  - **Manual mode** lets you set a fixed % share of the job per GPU, with the remainder auto-balanced across the rest
  - Adjustable safety ratio and fixed VRAM reserve so jobs don't run into out-of-memory errors
  - Settings persist between runs

## Requirements

- Python 3.x
- A working `transcribe-cli` binary from the Handy CLI project
- Vulkan drivers installed, if you want GPU acceleration
- (Fill in any Python package requirements here — e.g. `pip install -r requirements.txt`)

## Installation

```bash
cd C:\Users\User; if (!(Test-Path ".\HandyAidGUI")) { git clone https://github.com/joshuatUnlimited/HandyAidGUI.git }; cd .\HandyAidGUI; if (Test-Path ".\venv") { Remove-Item ".\venv" -Recurse -Force }; python -m venv venv; .\venv\Scripts\python.exe -m pip install --upgrade pip; .\venv\Scripts\python.exe -m pip install psutil tkinterdnd2; if (!(Test-Path ".\bin")) { New-Item -ItemType Directory ".\bin" | Out-Null }; if (!(Test-Path ".\models")) { New-Item -ItemType Directory ".\models" | Out-Null }; Write-Host "`n=== ENVIRONMENT ===" -ForegroundColor Cyan; .\venv\Scripts\python.exe --version; Write-Host "`n=== DEPENDENCIES ===" -ForegroundColor Cyan; .\venv\Scripts\python.exe -c "import tkinter,psutil,tkinterdnd2; print('Python dependencies OK')"; Write-Host "`n=== TRANSCRIBER ===" -ForegroundColor Cyan; if (Test-Path ".\bin\transcribe.exe") { .\bin\transcribe.exe --help } else { Write-Host "WARNING: Put your transcribe.exe in .\bin\ before testing transcription." -ForegroundColor Yellow }; Write-Host "`n=== LAUNCHING HANDYAIDGUI ===" -ForegroundColor Green; .\venv\Scripts\python.exe .\main.py
```

## Usage

1. Launch the app with `python main.py`
2. Point HandyAidGUI at your `transcribe-cli` binary and choose an input file
3. Open the GPU tab to review detected GPUs, enable/disable individual GPUs, and choose Auto or Manual workload weighting
4. Start the job

## GPU balancing notes

Each worker process is pinned to one physical Vulkan device explicitly (`--device 0`), rather than relying on `GGML_VK_VISIBLE_DEVICES` alone — the two controls are separate, and setting only the environment variable can leave multiple workers pointed at the same GPU. By default, work is split across enabled GPUs by available VRAM; manual per-GPU weights override this on a per-GPU basis.

This is **process-level load balancing** — each GPU worker gets its own full copy of the model and a share of the audio to process. It is not pooled VRAM and not tensor/model splitting across GPUs.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Unlicense](LICENSE)
