import os
import re
import shutil
from pathlib import Path

class CommandBuilderMixin:
    def validate_paths(self):
        model = Path(self.model_path_var.get().strip()).expanduser()
        audio = Path(self.audio_path_var.get().strip()).expanduser()
        binary = Path(self.binary_path_var.get().strip()).expanduser()
        if not model.is_file():
            raise ValueError("Please select a valid MOSS GGUF model file.")
        if not audio.is_file():
            raise ValueError("Please select a valid audio/video file.")

        binary_path = shutil.which(str(binary)) if not binary.is_file() else str(binary)
        if not binary_path:
            raise ValueError("Transcribe executable was not found. Select transcribe.exe (Windows) or the transcribe-cli binary.")
        if os.name != "nt" and not os.access(binary_path, os.X_OK):
            raise ValueError(f"'{binary_path}' was found but is not executable.")
        return str(binary_path), str(model.resolve()), str(audio.resolve())

    def build_command(self, binary, model, audio):
        # Match the documented MOSS quick-start shape first, while preventing
        # GUI-only optional settings from being passed to older binaries that
        # do not advertise the corresponding CLI flag.
        requested_backend = self.backend_var.get().strip().lower()
        cmd = [binary]
        if requested_backend in ("vulkan", "cpu"):
            cmd.extend(["--backend", requested_backend])
        cmd.extend(["-m", model])

        if self.speaker_mode_var.get() == "multi":
            if self.backend_supports("--diarize"):
                cmd.append("--diarize")
            else:
                raise ValueError("This transcriber executable does not advertise --diarize, which is required for multi-speaker mode.")
        else:
            if self.backend_supports("--no-diarize"):
                cmd.append("--no-diarize")
            elif self.backend_options:
                # Avoid silently enabling diarization when an older backend
                # lacks the explicit off switch.
                self.ui_queue.put(("log", "Warning: backend does not advertise --no-diarize; continuing without a speaker-mode flag.\n"))

        if self.backend_supports("--threads"):
            cmd.extend(["--threads", str(self.resolved_threads())])
        elif self.backend_options:
            self.ui_queue.put(("log", "Backend does not advertise --threads; using its native thread default.\n"))

        language_raw = self.language_var.get().strip().lower()
        if "(" in language_raw and language_raw.endswith(")"):
            language_raw = language_raw.rsplit("(", 1)[-1].rstrip(")").strip()
        if language_raw and language_raw != "auto" and re.fullmatch(r"[a-z]{2,3}", language_raw):
            if self.backend_supports("--language", "-l"):
                # -l is the documented MOSS short form and is widely supported.
                cmd.extend(["-l", language_raw])
            elif self.backend_options:
                self.ui_queue.put(("log", f"Backend does not advertise a language option; ignoring requested language '{language_raw}'.\n"))

        timestamps = self.timestamp_var.get().strip().lower()
        if timestamps in ("segment", "none"):
            if self.backend_supports("--timestamps"):
                cmd.extend(["--timestamps", timestamps])
            elif self.backend_options:
                self.ui_queue.put(("log", "Backend does not advertise --timestamps; leaving timestamp behavior at its default.\n"))

        cmd.append(audio)
        return cmd
