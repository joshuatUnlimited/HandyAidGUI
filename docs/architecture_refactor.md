# HandyAidGUI refactor

The original monolithic `moss_transcriber_gui.py` is used as the behavioral reference.
The application is split into mixins/modules while preserving the existing GUI state
and workflow contract.

## Responsibilities

- `app/gui/`: Tk/ttk presentation, layout, controls, scrolling and dialogs.
- `app/engine/`: executable validation, CLI construction and transcription lifecycle.
- `app/audio/`: FFmpeg preparation and duration detection.
- `app/models/`: GGUF metadata and model compatibility.
- `app/backends/`: Vulkan/CPU/Auto backend selection and environment.
- `app/parsing/`: MOSS segment parsing and progress extraction.
- `app/output/`: TXT/Markdown/SRT/JSON/raw output serialization.
- `app/config/`: persistent application settings and constants.

`main.py` is intentionally thin and only boots the application.
