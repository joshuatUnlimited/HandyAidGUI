import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path


class AudioConverterMixin:
    def prepare_audio(self, audio_path):
        path = Path(audio_path)
        needs_convert = path.suffix.lower() != ".wav"

        if not needs_convert:
            try:
                with wave.open(str(path), "rb") as wav:
                    needs_convert = (
                        wav.getnchannels() != 1
                        or wav.getframerate() != 16000
                    )
            except (wave.Error, OSError):
                needs_convert = True

        if not needs_convert:
            return str(path), None

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "This input needs conversion to 16 kHz mono WAV, "
                "but ffmpeg was not found on PATH."
            )

        fd, temp_path = tempfile.mkstemp(prefix="moss_", suffix=".wav")
        os.close(fd)

        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(path),
            "-ar",
            "16000",
            "-ac",
            "1",
            temp_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise RuntimeError(
                f"ffmpeg conversion failed:\n{result.stderr[-1200:]}"
            )
        return temp_path, temp_path
