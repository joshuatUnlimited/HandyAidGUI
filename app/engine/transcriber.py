import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

class TranscriptionEngineMixin:
    def start_transcription(self):
        if self.job_active:
            return
        try:
            binary, model, audio = self.validate_paths()
        except ValueError as exc:
            messagebox.showerror("Cannot Start", str(exc))
            return

        output_text = self.output_path_var.get().strip()
        if not output_text:
            self.set_default_output()
            output_text = self.output_path_var.get().strip()
        if not output_text:
            messagebox.showerror("Output Error", "Choose an output destination first.")
            return

        model_info = self.inspect_model_compatibility(model)
        self.model_info = model_info
        self.compatible_model_candidates = model_info.get("candidates", [])
        self.update_model_status()
        if model_info.get("compatible") is False:
            details = model_info.get("reason", "The selected model is incompatible with this backend.")
            if self.compatible_model_candidates:
                details += "\n\nCompatible MOSS model(s) found in the same folder:\n" + "\n".join(self.compatible_model_candidates[:6])
            details += ("\n\nThe GUI stopped before FFmpeg conversion, so your 75-minute recording was not processed unnecessarily.")
            self.append_log("\n--- Model compatibility failure ---\n" + details + "\n")
            self.status_var.set("Incompatible model")
            messagebox.showerror("Incompatible MOSS model", details)
            return

        output = Path(output_text).expanduser()
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Output Error", str(exc))
            return

        self.append_log("\n--- Backend preflight ---\n")
        help_text = self.probe_backend_help(binary)
        if help_text and help_text.lower().startswith("unable to query"):
            self.append_log(help_text + "\n")
        elif help_text:
            self.append_log("Backend help detected; using documented MOSS command syntax.\n")

        backend_ok, backend_message = self.verify_requested_backend(binary)
        self.append_log(backend_message + "\n")
        if not backend_ok:
            self.status_var.set("Vulkan unavailable")
            self.append_log("\n--- GPU backend unavailable ---\n" + backend_message + "\n")
            messagebox.showerror("Vulkan backend unavailable", backend_message + "\n\nSee the Process Log for details.")
            return

        try:
            env_preview = self.vulkan_environment()
            cmd = self.build_command(binary, model, audio)
        except ValueError as exc:
            messagebox.showerror("Backend Configuration", str(exc))
            self.status_var.set("Configuration error")
            return
        self.last_command = cmd
        if self.backend_var.get().strip().lower() == "vulkan":
            devices = env_preview.get("GGML_VK_VISIBLE_DEVICES", "")
            self.append_log(f"Vulkan visible devices: {devices or 'backend default'}\n")
            if devices and "," in devices:
                self.append_log("Multi-device Vulkan visibility is enabled; the backend may split work across visible adapters depending on its ggml build/runtime.\n")
        self.current_output_path = output
        self.current_audio_path = audio
        self.last_segments = []
        self.last_raw_output = ""
        self.cancel_requested.clear()
        self.job_active = True
        self.started_at = time.monotonic()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress_var.set(0)
        self.set_transcript_text("")
        self.status_var.set("Starting…")
        self.sidebar_status.config(text="Starting…")
        self.header_badge.config(text="● STARTING")
        self.append_log("\n--- Starting transcription ---\n")
        self.append_log(f"Mode: {'Multi-speaker' if self.speaker_mode_var.get() == 'multi' else 'No diarization'}\n")
        self.append_log(f"Input: {audio}\n")
        self.append_log(f"Output: {output}\n")
        self.set_command_display(self.format_command_for_display(cmd))
        if not self.current_audio_duration:
            self.current_audio_duration = self.get_audio_duration(audio)
        self.refresh_summaries()
        if self.current_audio_duration:
            self.append_log(f"Audio duration: {self.format_seconds(self.current_audio_duration)}\n")
        else:
            self.append_log("Could not determine audio duration; showing an indeterminate progress bar.\n")
            self.ui_queue.put(("indeterminate", None))

        self.save_settings()
        self.worker = threading.Thread(target=self.run_transcription, args=(cmd, audio), daemon=True)
        self.worker.start()

    def run_transcription(self, cmd, audio_path):
        raw_output = []
        parsed_segments = []
        pending_log = []
        pending_segments = None
        last_ui_flush = time.monotonic()
        temp_wav = None
        started = time.monotonic()
        try:
            prepared_audio, temp_wav = self.prepare_audio(audio_path)
            if prepared_audio != cmd[-1]:
                cmd = list(cmd)
                cmd[-1] = prepared_audio
                self.ui_queue.put(("command", self.format_command_for_display(cmd)))
                self.ui_queue.put(("log", "Converted input to 16 kHz mono WAV for MOSS.\n"))

            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                env=self.vulkan_environment(),
            )
            self.ui_queue.put(("status", "Transcribing…"))
            assert self.process.stdout is not None
            for line in iter(self.process.stdout.readline, ""):
                if line:
                    raw_output.append(line)
                    pending_log.append(line)
                    backend_match = re.search(r"\bbackend\s*:\s*(.+)", line, re.IGNORECASE)
                    if backend_match:
                        detected_backend = backend_match.group(1).strip()
                        self.ui_queue.put(("status", f"Running on {detected_backend}"))
                        self.ui_queue.put(("backend_runtime", detected_backend))
                    parsed_segments = self.update_segments(parsed_segments, line)
                    pending_segments = list(parsed_segments)
                    now = time.monotonic()
                    # Update the GUI at most ~10 times/sec instead of once per backend line.
                    # This materially reduces Tk overhead on verbose MOSS output.
                    if now - last_ui_flush >= 0.10:
                        self.ui_queue.put(("log", "".join(pending_log)))
                        self.ui_queue.put(("segments", pending_segments))
                        pending_log.clear()
                        pending_segments = None
                        last_ui_flush = now
                    self.update_progress_from_line(line)
                if self.cancel_requested.is_set():
                    self.terminate_process()
                    break

            if pending_log:
                self.ui_queue.put(("log", "".join(pending_log)))
            if pending_segments is not None:
                self.ui_queue.put(("segments", pending_segments))
            elif parsed_segments:
                self.ui_queue.put(("segments", list(parsed_segments)))

            rc = self.process.wait()
            cancelled = self.cancel_requested.is_set()
            elapsed = time.monotonic() - started
            output_text = "".join(raw_output)
            if rc == 0 and not cancelled:
                parsed_segments = self.parse_final_segments(output_text)
            self.ui_queue.put(("finished", (rc, cancelled, output_text, parsed_segments, elapsed)))
        except FileNotFoundError as exc:
            self.ui_queue.put(("finished", (None, False, "", [], time.monotonic() - started, f"Executable error: {exc}")))
        except Exception as exc:
            self.ui_queue.put(("finished", (None, False, "", [], time.monotonic() - started, f"Unexpected error: {exc}")))
        finally:
            self.process = None
            if temp_wav:
                self.cleanup_temp_file(temp_wav)

    def cleanup_temp_file(self, path):
        for _ in range(attempts):
            try:
                os.unlink(path)
                return
            except OSError:
                time.sleep(delay)

    def terminate_process(self):
        process = self.process
        if not process or process.poll() is not None:
            return
        try:
            process.terminate()
            deadline = time.monotonic() + 2.0
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if process.poll() is None:
                process.kill()
        except OSError:
            pass

    def cancel_transcription(self):
        if not self.job_active:
            return
        self.cancel_requested.set()
        self.status_var.set("Stopping…")
        self.sidebar_status.config(text="Stopping…")
        self.append_log("\n--- Stop requested ---\n")
        threading.Thread(target=self.terminate_process, daemon=True).start()

    def finish_job(self, rc, cancelled, raw_output, segments, elapsed, error_message=None):
        self.job_active = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress.stop()
        self.progress.configure(mode="determinate")
        if self.current_audio_duration and rc == 0 and not cancelled:
            self.progress_var.set(100)

        self.last_segments = segments
        self.last_raw_output = raw_output
        self.set_transcript_text(self.extract_transcript(raw_output, segments))
        self.elapsed_var.set(self.format_seconds(elapsed, always_hours=True))
        self.sidebar_elapsed.config(text=f"Elapsed {self.elapsed_var.get()}")

        if error_message:
            self.append_log(f"\n--- {error_message} ---\n")
            self.status_var.set("Failed")
            messagebox.showerror("Transcription Error", error_message)
            return
        if cancelled:
            self.append_log("\n--- Transcription stopped ---\n")
            self.status_var.set("Stopped")
            return
        if rc == 0:
            if self.speaker_mode_var.get() == "multi" and segments and not any(s.get("speaker") for s in segments):
                self.append_log("Warning: multi-speaker mode was requested but no speaker tags were found in the output.\n")
            try:
                self.write_output(raw_output, segments)
                self.append_log(f"\n--- Transcription complete ---\nSaved: {self.current_output_path}\n")
                self.status_var.set("Complete")
            except OSError as exc:
                self.append_log(f"\n--- Transcription finished, but output save failed: {exc} ---\n")
                self.status_var.set("Output save failed")
                messagebox.showerror("Output Error", str(exc))
        else:
            self.last_failure_details = "\n".join(raw_output.splitlines()[-40:]).strip()
            self.append_log(f"\n--- Process failed with exit code {rc} ---\n")
            if self.last_failure_details:
                self.append_log("\n--- Backend error tail ---\n" + self.last_failure_details + "\n")
            self.append_log("\nCommand: " + self.format_command_for_display(self.last_command) + "\n")
            self.status_var.set(f"Failed (exit {rc})")
            detail = self.last_failure_details or "No backend diagnostics were returned."
            if len(detail) > 3500:
                detail = detail[-3500:]
            messagebox.showerror(
                "Transcription Failed",
                f"The transcriber exited with code {rc}.\n\nBackend diagnostics:\n{detail}\n\nThe full diagnostics remain in Process Log.",
            )
