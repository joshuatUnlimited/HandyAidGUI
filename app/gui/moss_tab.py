# app/gui/moss_tab.py

import os
import re
import subprocess
import threading
import tempfile
import json
import wave
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

import numpy as np
import torch

from app.config.settings import SUPPORTED_AUDIO


class MossTab:
    """MOSS audio chunking and transcription tab (uses transcribe.exe)."""

    def __init__(self, parent, app_window):
        self.parent = parent
        self.app = app_window
        self.file_path = None
        self.is_processing = False
        self._cancel = False
        self.build_ui()

    # ---------- UI ----------
    def build_ui(self):
        file_frame = ttk.LabelFrame(self.parent, text="Audio File", padding=10)
        file_frame.pack(fill="x", pady=(0, 12))
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Entry(file_frame, textvariable=self.file_var, state="readonly").pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side="right")

        chunk_frame = ttk.LabelFrame(self.parent, text="Chunking Parameters", padding=10)
        chunk_frame.pack(fill="x", pady=(0, 12))
        duration_row = ttk.Frame(chunk_frame)
        duration_row.pack(fill="x", pady=2)
        ttk.Label(duration_row, text="Chunk duration (seconds):").pack(side="left", padx=(0, 10))
        self.chunk_duration = tk.IntVar(value=60)
        ttk.Spinbox(duration_row, from_=10, to=600, textvariable=self.chunk_duration, width=8).pack(side="left")
        ttk.Label(duration_row, text="(recommended ≤ 300)").pack(side="left", padx=(10, 0))

        policy_row = ttk.Frame(chunk_frame)
        policy_row.pack(fill="x", pady=2)
        ttk.Label(policy_row, text="Incomplete chunk policy:").pack(side="left", padx=(0, 10))
        self.truncate_policy = tk.StringVar(value="keep")
        ttk.Radiobutton(policy_row, text="Keep", variable=self.truncate_policy, value="keep").pack(side="left")
        ttk.Radiobutton(policy_row, text="Drop", variable=self.truncate_policy, value="drop").pack(side="left", padx=(10, 0))

        btn_frame = ttk.Frame(self.parent)
        btn_frame.pack(fill="x", pady=8)
        self.process_btn = ttk.Button(btn_frame, text="Start Transcription", command=self.start_processing, style="Accent.TButton")
        self.process_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_processing, state="disabled", style="Danger.TButton")
        self.stop_btn.pack(side="left")

        self.progress = ttk.Progressbar(self.parent, mode="determinate")
        self.progress.pack(fill="x", pady=6)

        log_frame = ttk.LabelFrame(self.parent, text="Processing Log", padding=8)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.log_text = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", " ".join(f"*.{ext}" for ext in SUPPORTED_AUDIO))]
        )
        if path:
            self.file_path = path
            self.file_var.set(os.path.basename(path))
            self.log(f"Selected file: {path}")

    def log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ---------- Control ----------
    def start_processing(self):
        if not self.file_path or not os.path.exists(self.file_path):
            messagebox.showerror("Error", "Please select a valid audio file first.")
            return
        if self.is_processing:
            return

        binary = self.app.binary_path_var.get().strip()
        model = self.app.model_path_var.get().strip()
        if not binary or not Path(binary).is_file():
            messagebox.showerror("Error", "Please select a valid transcribe.exe in the Engine tab.")
            return
        if not model or not Path(model).is_file():
            messagebox.showerror("Error", "Please select a valid MOSS model in the Engine tab.")
            return

        self.is_processing = True
        self.process_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress["value"] = 0
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.log("=" * 50)
        self.log(f"Starting processing: {os.path.basename(self.file_path)}")
        self.log(f"Chunk duration: {self.chunk_duration.get()} seconds")
        self.log(f"Incomplete chunk policy: {self.truncate_policy.get()}")

        self.thread = threading.Thread(target=self._process, daemon=True)
        self.thread.start()

    def stop_processing(self):
        self.log("⏹ Stopping... (will exit after current chunk)")
        self._cancel = True

    # ---------- Pipeline ----------
    def _process(self):
        try:
            self._cancel = False
            transcript = self._run_pipeline()
            if not self._cancel:
                self.parent.after(0, self._on_finished, transcript)
            else:
                self.parent.after(0, self._on_cancelled)
        except Exception as e:
            self.parent.after(0, self._on_error, str(e))

    def _run_pipeline(self):
        self.parent.after(0, lambda: self.log("Loading audio via ffmpeg..."))
        audio, sr = self._load_audio_ffmpeg(self.file_path)
        if audio is None:
            return ""

        total_samples = len(audio)
        chunk_samples = int(self.chunk_duration.get() * sr)
        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        self.parent.after(0, lambda: self.log(
            f"Audio duration: {total_samples/sr:.1f}s, splitting into {num_chunks} chunks"
        ))

        all_segments = []
        speaker_map = {}
        next_global_id = 1

        for i in range(num_chunks):
            if self._cancel:
                return ""

            start = i * chunk_samples
            end = min(start + chunk_samples, total_samples)
            chunk_audio = audio[start:end]

            if self.truncate_policy.get() == "drop" and len(chunk_audio) < chunk_samples:
                self.parent.after(0, lambda: self.log(
                    f"Dropping final incomplete chunk (< {self.chunk_duration.get()}s)"
                ))
                break

            self.parent.after(0, lambda idx=i+1, total=num_chunks:
                              self.log(f"Processing chunk {idx}/{total}..."))

            result = self._transcribe_chunk(chunk_audio, sr)

            offset = start / sr
            for seg in result:
                seg["start"] += offset
                seg["end"] += offset

            for seg in result:
                local_id = seg.get("speaker", "S01")
                if local_id not in speaker_map:
                    global_id = f"G{next_global_id:02d}"
                    speaker_map[local_id] = global_id
                    next_global_id += 1
                seg["speaker"] = speaker_map[local_id]

            all_segments.extend(result)

            progress = int((i + 1) / num_chunks * 100)
            self.parent.after(0, lambda p=progress: self.progress.configure(value=p))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_segments.sort(key=lambda x: x["start"])
        transcript = self._format_transcript(all_segments)
        return transcript

    # ---------- Audio loading (ffmpeg) ----------
    def _load_audio_ffmpeg(self, filepath, target_sr=16000):
        """Load audio file using ffmpeg, return (numpy array, sample_rate)."""
        cmd = [
            "ffmpeg",
            "-i", filepath,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(target_sr),
            "-ac", "1",
            "-"
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            raw, _ = proc.communicate()
            if proc.returncode != 0:
                self.parent.after(0, lambda: self.log("Error: ffmpeg failed to decode audio."))
                return None, target_sr
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return audio, target_sr
        except Exception as e:
            self.parent.after(0, lambda: self.log(f"ffmpeg error: {str(e)}"))
            return None, target_sr

    # ---------- Transcribe a single chunk ----------
    def _transcribe_chunk(self, audio, sr):
        """
        Transcribe one audio chunk using the selected transcribe.exe.
        Returns a list of dicts with keys: start, end, speaker, text.
        """
        # Write chunk to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # Write 16-bit PCM WAV
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                int_audio = (audio * 32767).astype(np.int16).tobytes()
                wf.writeframes(int_audio)

            # Build command from main app settings
            binary = self.app.binary_path_var.get().strip()
            model = self.app.model_path_var.get().strip()
            backend = self.app.backend_var.get().lower()   # 'vulkan', 'auto', or 'cpu'
            threads = self.app.resolved_threads()

            # Language: extract ISO code from "English (en)" or use raw string if just "en"
            raw_lang = self.app.language_var.get().strip()
            lang_code = raw_lang
            # Try to extract code inside parentheses, e.g., "English (en)" -> "en"
            match = re.search(r'\((\w+)\)', raw_lang)
            if match:
                lang_code = match.group(1)
            elif raw_lang.lower() == "auto":
                lang_code = "auto"
            # else keep as is (may be "en" or "de" etc.)

            # Timestamps: map GUI values to binary values
            ts_setting = self.app.timestamp_var.get()
            if ts_setting == "Segment":
                ts_value = "segment"
            elif ts_setting == "None":
                ts_value = "none"
            else:  # "Auto"
                ts_value = "auto"

            # Build command
            # The binary expects: transcribe-cli [options] audio.wav
            # Options: -m model, --backend, --threads, -l language, --timestamps, --diarize, etc.
            cmd = [
                binary,
                "-m", model,
                "--backend", backend,
                "--threads", str(threads),
                "-l", lang_code,
                "--timestamps", ts_value,
                "--diarize",          # enable speaker diarization for MOSS
                tmp_path              # positional argument: input audio file
            ]

            self.parent.after(0, lambda: self.log(f"Running: {' '.join(cmd)}"))
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            stdout, stderr = proc.communicate(timeout=300)   # 5 min per chunk

            if stdout:
                snippet = stdout[:500] + ("..." if len(stdout) > 500 else "")
                self.parent.after(0, lambda: self.log(f"STDOUT (first 500):\n{snippet}"))
            if stderr:
                self.parent.after(0, lambda: self.log(f"STDERR:\n{stderr}"))

            if proc.returncode != 0:
                self.parent.after(0, lambda: self.log(f"Transcription failed with code {proc.returncode}"))
                return self._fallback_segment(audio, sr, text=f"Error: {stderr}")

            # Try to parse as JSON if stdout starts with '{' or '[' (maybe the binary outputs JSON)
            if stdout.strip().startswith(('{', '[')):
                segments = self._parse_json_output(stdout)
                if segments:
                    return segments

            # Fallback: treat entire stdout as one segment
            if stdout.strip():
                return [{
                    "start": 0.0,
                    "end": len(audio) / sr,
                    "speaker": "S01",
                    "text": stdout.strip()
                }]
            else:
                self.parent.after(0, lambda: self.log("No output from transcribe."))
                return self._fallback_segment(audio, sr, text="(Empty output)")

        except subprocess.TimeoutExpired:
            self.parent.after(0, lambda: self.log("Transcription timed out for this chunk."))
            return self._fallback_segment(audio, sr, text="(Timeout)")
        except Exception as e:
            self.parent.after(0, lambda: self.log(f"Error in _transcribe_chunk: {str(e)}"))
            return self._fallback_segment(audio, sr, text=f"(Error: {e})")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _parse_json_output(self, text):
        """Parse the transcribe.exe JSON output into list of segments."""
        try:
            data = json.loads(text)
            # Assume either a dict with "segments" or a list directly
            if isinstance(data, dict):
                if "segments" in data:
                    segments = data["segments"]
                else:
                    # maybe the whole dict is a segment?
                    return [{
                        "start": data.get("start", 0.0),
                        "end": data.get("end", 0.0),
                        "speaker": data.get("speaker", "S01"),
                        "text": data.get("text", "").strip(),
                    }] if "text" in data else None
            elif isinstance(data, list):
                segments = data
            else:
                return None

            result = []
            for seg in segments:
                if isinstance(seg, dict):
                    result.append({
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "speaker": seg.get("speaker", "S01"),
                        "text": seg.get("text", "").strip(),
                    })
            return result if result else None
        except json.JSONDecodeError:
            return None

    def _fallback_segment(self, audio, sr, text=None):
        """Return a single segment covering the whole chunk."""
        duration = len(audio) / sr
        return [{
            "start": 0.0,
            "end": duration,
            "speaker": "S01",
            "text": text or f"[Placeholder] This chunk is {duration:.1f}s long"
        }]

    # ---------- Formatting and output ----------
    def _format_transcript(self, segments):
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip()
            if text:
                lines.append(f"[{speaker}] {text}")
        return "\n".join(lines) if lines else "(No transcription content)"

    def _on_finished(self, transcript):
        self.is_processing = False
        self.process_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.log("=" * 50)
        self.log("✅ Processing complete!")
        self.log("\n--- Final Transcript ---\n")
        self.log(transcript)
        self._save_result(transcript)

    def _on_cancelled(self):
        self.is_processing = False
        self.process_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.log("⛔ Cancelled by user")

    def _on_error(self, error_msg):
        self.is_processing = False
        self.process_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.log(f"❌ Error: {error_msg}")
        messagebox.showerror("Processing Error", error_msg)

    def _save_result(self, transcript):
        if not transcript or transcript == "(No transcription content)":
            return
        output_dir = os.path.join(os.path.dirname(self.file_path), "transcripts")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_moss.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        self.log(f"📁 Result saved to: {output_path}")