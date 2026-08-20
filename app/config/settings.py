import json
from pathlib import Path

APP_NAME = "MOSS Transcriber"
APP_SUBTITLE = "Offline speech-to-text workstation"
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
            ("vulkan_devices", self.vulkan_devices_var),
        ]:
            value = data.get(key)
            if value is not None:
                var.set(value)

        threads = data.get("threads")
        if isinstance(threads, int) and threads > 0:
            self.threads_var.set(threads)

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
