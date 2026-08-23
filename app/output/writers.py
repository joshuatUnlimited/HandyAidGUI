import json
from pathlib import Path

class OutputWriterMixin:
    def write_output(self, raw_output, segments):
        output = Path(self.current_output_path)
        fmt = self.output_format_var.get()
        transcript = self.extract_transcript(raw_output, segments)
        if fmt == "TXT":
            output.write_text(transcript + "\n", encoding="utf-8")
        elif fmt == "MARKDOWN":
            output.write_text(self.to_markdown(transcript, segments), encoding="utf-8")
        elif fmt == "RAW LOG":
            output.write_text(raw_output, encoding="utf-8")
        elif fmt == "JSON":
            payload = {
                "model": Path(self.model_path_var.get()).name,
                "audio": Path(self.audio_path_var.get()).name,
                "speaker_mode": "multi-speaker" if self.speaker_mode_var.get() == "multi" else "no-diarization",
                "transcript": transcript,
                "segments": segments,
            }
            output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        elif fmt == "SRT":
            output.write_text(self.to_srt(segments, transcript), encoding="utf-8")

    def to_markdown(self, transcript, segments):
        audio_name = Path(self.audio_path_var.get()).name if self.audio_path_var.get().strip() else "Unknown audio"
        model_name = Path(self.model_path_var.get()).name if self.model_path_var.get().strip() else "Unknown model"
        mode = "Multi-speaker diarization" if self.speaker_mode_var.get() == "multi" else "Single-speaker"
        language = self.language_var.get().strip() or "Auto"
        duration = self.current_audio_duration

        lines = [
            "# Transcription",
            "",
            f"- **Source:** `{audio_name}`",
            f"- **Model:** `{model_name}`",
            f"- **Mode:** {mode}",
            f"- **Language:** {language}",
        ]
        if duration:
            lines.append(f"- **Duration:** {self.format_seconds(duration, always_hours=True)}")
        lines.extend(["", "## Transcript", ""])

        if segments:
            for seg in segments:
                start = self.format_seconds(seg["start"], always_hours=True)
                end = self.format_seconds(seg["end"], always_hours=True)
                speaker = f"**{seg['speaker']}** — " if seg.get("speaker") else ""
                lines.append(f"**[{start} → {end}]** {speaker}{seg['text']}")
                lines.append("")
        else:
            lines.append(transcript or "")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def to_srt(segments, transcript):
        if not segments:
            return transcript + "\n"
        parts = []
        for index, seg in enumerate(segments, 1):
            speaker = f"{seg['speaker']}: " if seg.get("speaker") else ""
            parts.append(f"{index}\n{OutputWriterMixin.srt_time(seg['start'])} --> {OutputWriterMixin.srt_time(seg['end'])}\n{speaker}{seg['text']}\n")
        return "\n".join(parts)

    @staticmethod
    def srt_time(seconds):
        total_ms = max(0, int(round(seconds * 1000)))
        hours, rem = divmod(total_ms, 3_600_000)
        minutes, rem = divmod(rem, 60_000)
        secs, ms = divmod(rem, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
