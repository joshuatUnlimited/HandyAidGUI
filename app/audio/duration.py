import shutil
import subprocess
import wave

class AudioDurationMixin:
    @staticmethod
    def get_audio_duration(audio_path):
        try:
            with wave.open(audio_path, "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                return frames / rate if rate else None
        except (wave.Error, OSError):
            ffprobe = shutil.which("ffprobe")
            if not ffprobe:
                return None
            try:
                result = subprocess.run(
                    [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=10,
                )
                return float(result.stdout.strip())
            except (ValueError, OSError, subprocess.SubprocessError):
                return None

    @staticmethod
    def format_seconds(seconds, always_hours=False):
        total = int(max(0, round(seconds)))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours or always_hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
