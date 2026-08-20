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

        binary_path = (
            shutil.which(str(binary))
            if not binary.is_file()
            else str(binary)
        )
        if not binary_path:
            raise ValueError(
                "Transcribe executable was not found. "
                "Select transcribe.exe (Windows) or the transcribe-cli binary."
            )

        if os.name != "nt" and not os.access(binary_path, os.X_OK):
            raise ValueError(
                f"'{binary_path}' was found but is not executable."
            )

        return (
            str(binary_path),
            str(model.resolve()),
            str(audio.resolve()),
        )

    def _vulkan_runtime_device_index(self):
        """
        Resolve the process-local transcribe device index.

        GGML_VK_VISIBLE_DEVICES uses physical Vulkan indices. transcribe-cli's
        --device uses the process-local transcribe device registry. When the
        environment exposes N Vulkan adapters, those Vulkan adapters appear in
        the runtime registry in their visible order, followed by the CPU.

        Therefore:
          physical selection "1" only -> runtime Vulkan device 0
          physical selection "0,1"     -> runtime Vulkan devices 0 and 1

        HandyAidGUI intentionally uses the FIRST selected Vulkan adapter for a
        normal single-process run. Multi-GPU workers each expose exactly one
        physical adapter and therefore also use runtime device 0.
        """
        try:
            selected = self.vulkan_device_ids()
        except (AttributeError, ValueError):
            return None

        if not selected:
            return None

        return 0

    def build_command(self, binary, model, audio):
        # Match the documented MOSS command shape while only passing options
        # advertised by the installed executable.
        requested_backend = self.backend_var.get().strip().lower()
        cmd = [binary]

        if requested_backend in ("vulkan", "cpu"):
            cmd.extend(["--backend", requested_backend])

        # CRITICAL: Vulkan visibility and transcribe device selection are two
        # separate mechanisms. GGML_VK_VISIBLE_DEVICES controls physical
        # visibility; --device selects the process-local compute device.
        #
        # Without --device, the runtime may automatically choose another
        # device (including CPU fallback), which made the prior GUI appear to
        # use neither selected GPU.
        if (
            requested_backend == "vulkan"
            and self.backend_supports("--device")
        ):
            runtime_device = self._vulkan_runtime_device_index()
            if runtime_device is not None:
                cmd.extend(["--device", str(runtime_device)])

        cmd.extend(["-m", model])

        if self.speaker_mode_var.get() == "multi":
            if self.backend_supports("--diarize"):
                cmd.append("--diarize")
            else:
                raise ValueError(
                    "This transcriber executable does not advertise "
                    "--diarize, which is required for multi-speaker mode."
                )
        else:
            if self.backend_supports("--no-diarize"):
                cmd.append("--no-diarize")
            elif self.backend_options:
                # Avoid silently enabling diarization when an older backend
                # lacks the explicit off switch.
                self.ui_queue.put(
                    (
                        "log",
                        "Warning: backend does not advertise --no-diarize; "
                        "continuing without a speaker-mode flag.\n",
                    )
                )

        if self.backend_supports("--threads"):
            cmd.extend(
                ["--threads", str(self.resolved_threads())]
            )
        elif self.backend_options:
            self.ui_queue.put(
                (
                    "log",
                    "Backend does not advertise --threads; "
                    "using its native thread default.\n",
                )
            )

        language_raw = self.language_var.get().strip().lower()
        if "(" in language_raw and language_raw.endswith(")"):
            language_raw = (
                language_raw.rsplit("(", 1)[-1]
                .rstrip(")")
                .strip()
            )

        if (
            language_raw
            and language_raw != "auto"
            and re.fullmatch(r"[a-z]{2,3}", language_raw)
        ):
            if self.backend_supports("--language", "-l"):
                # -l is the documented MOSS short form.
                cmd.extend(["-l", language_raw])
            elif self.backend_options:
                self.ui_queue.put(
                    (
                        "log",
                        "Backend does not advertise a language option; "
                        f"ignoring requested language '{language_raw}'.\n",
                    )
                )

        timestamps = self.timestamp_var.get().strip().lower()
        if timestamps in ("segment", "none"):
            if self.backend_supports("--timestamps"):
                cmd.extend(["--timestamps", timestamps])
            elif self.backend_options:
                self.ui_queue.put(
                    (
                        "log",
                        "Backend does not advertise --timestamps; "
                        "leaving timestamp behavior at its default.\n",
                    )
                )

        cmd.append(audio)
        return cmd
