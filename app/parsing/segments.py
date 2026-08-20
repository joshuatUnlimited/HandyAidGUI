import re


class ParsingMixin:
    SEGMENT_RE = re.compile(
        r"\[(?P<start>\d+(?:\.\d+)?)\](?:\[(?P<speaker>S\d+)\])?"
        r"(?P<text>.*?)(?=\[(?P<end>\d+(?:\.\d+)?)\])",
        re.DOTALL,
    )
    FINAL_SENTINEL = "\n[999999999]"

    def parse_final_segments(self, raw_output):
        return self.update_segments([], raw_output + self.FINAL_SENTINEL)

    def update_segments(self, current, line):
        text = line
        matches = list(self.SEGMENT_RE.finditer(text))
        if not matches:
            return current

        result = list(current)
        for match in matches:
            clean_text = re.sub(r"\s+", " ", match.group("text")).strip()
            if not clean_text:
                continue

            segment = {
                "start": float(match.group("start")),
                "end": float(match.group("end")),
                "speaker": match.group("speaker"),
                "text": clean_text,
            }
            if result and self.same_segment(result[-1], segment):
                result[-1] = segment
            else:
                result.append(segment)
        return result

    @staticmethod
    def same_segment(a, b):
        return (
            abs(a["start"] - b["start"]) < 0.001
            and abs(a["end"] - b["end"]) < 0.001
        )

    def extract_transcript(self, raw_output, segments):
        if segments:
            return "\n".join(segment["text"] for segment in segments).strip()

        lines = []
        for line in raw_output.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(
                ("[", "main", "system", "loading", "backend")
            ):
                continue
            lines.append(stripped)
        return "\n".join(lines).strip()

    def update_progress_from_line(self, line):
        if not self.current_audio_duration:
            return

        end_times = [
            float(m.group(1))
            for m in re.finditer(r"\[(\d+(?:\.\d+)?)\]", line)
        ]
        if end_times:
            progress = min(
                100.0,
                max(0.0, max(end_times) / self.current_audio_duration * 100.0),
            )
            self.ui_queue.put(("determinate", None))
            self.ui_queue.put(("progress", progress))
            self.ui_queue.put(("status", f"Transcribing… {progress:.0f}%"))
