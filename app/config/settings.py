import json
import os
from pathlib import Path

from app.backends.gpu_manager import VulkanGPUManager

APP_NAME = "HandyAid"
APP_SUBTITLE = "Offline transcription workstation — any GGUF model"
CONFIG_FILE = Path.home() / ".moss_handy_transcriber.json"
SUPPORTED_AUDIO = [
    ("Audio / Video", "*.wav *.mp3 *.flac *.ogg *.m4a *.wma *.aac *.mp4 *.webm"),
    ("WAV Audio", "*.wav"), ("MP3 Audio", "*.mp3"), ("FLAC Audio", "*.flac"),
    ("M4A Audio", "*.m4a"), ("MP4 Video", "*.mp4"), ("All Files", "*.*"),
]

class SettingsMixin:
    def load_settings(self):
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return

        for key, var in [
            ("model", self.model_path_var),
            ("binary", self.binary_path_var),
            ("audio", self.audio_path_var),
            ("output", self.output_path_var),
            ("output_format", self.output_format_var),
            ("language", self.language_var),
            ("timestamps", self.timestamp_var),
            ("speaker_mode", self.speaker_mode_var),
            ("backend", self.backend_var),
        ]:
            value = data.get(key)
            if value is not None:
                var.set(value)

        # vulkan_devices gets format-only validation here (pure parsing,
        # no hardware probe — settings load must stay cheap and
        # dependency-free). Whether the IDs correspond to real, currently
        # present adapters is re-checked against actual hardware right
        # before a transcription starts (see transcriber.py), and the GPU
        # panel (gpu_panel.py, when wired up via gpu_manager.GPUManager)
        # populates this field from real hardware in the first place —
        # this is just a backstop against a hand-edited or otherwise
        # malformed config value crashing settings load or silently
        # carrying garbage forward.
        vulkan_devices = data.get("vulkan_devices")
        if isinstance(vulkan_devices, str):
            try:
                VulkanGPUManager.parse_ids(vulkan_devices)
            except ValueError:
                pass  # malformed (e.g. hand-edited); leave the field at its default
            else:
                self.vulkan_devices_var.set(vulkan_devices)

        threads = data.get("threads")
        max_threads = max(1, os.cpu_count() or 16)
        if isinstance(threads, int) and 0 < threads <= max_threads:
            self.threads_var.set(threads)
        elif isinstance(threads, int) and threads > max_threads:
            # Stale/corrupted config from before the upper-bound clamp existed
            # (or a hand-edited value). Fall back to a safe default instead of
            # silently handing an oversized --threads value to the backend.
            self.threads_var.set(max_threads)

    def save_settings(self):
        data = {
            "model": self.model_path_var.get(),
            "binary": self.binary_path_var.get(),
            "audio": self.audio_path_var.get(),
            "output": self.output_path_var.get(),
            "output_format": self.output_format_var.get(),
            "threads": int(self.resolved_threads()),
            "speaker_mode": self.speaker_mode_var.get(),
            "language": self.language_var.get(),
            "timestamps": self.timestamp_var.get(),
            "backend": self.backend_var.get(),
            "vulkan_devices": self.vulkan_devices_var.get().strip(),
        }
        try:
            CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass
