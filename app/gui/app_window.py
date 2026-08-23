import os
import queue
import re
import shutil
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAVE_DND = True
except ImportError:
    HAVE_DND = False
    DND_FILES = None
    TkinterDnD = None

from app.config.settings import SUPPORTED_AUDIO
from app.gui.moss_tab import MossTab   # <-- Import added here


class AppWindowMixin:
    def build_ui(self):
        shell = ttk.Frame(self.root)
        shell.pack(fill="both", expand=True)
        shell.rowconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)

        canvas = tk.Canvas(
            shell,
            background=self.BG,
            highlightthickness=0,
            borderwidth=0,
        )
        canvas.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=vscroll.set)
        self.main_canvas = canvas

        page = ttk.Frame(canvas, padding=18)
        self.scroll_page = page
        window_id = canvas.create_window((0, 0), window=page, anchor="nw")

        def update_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_page(event):
            canvas.itemconfigure(window_id, width=event.width)
            update_scrollregion()

        page.bind("<Configure>", update_scrollregion)
        canvas.bind("<Configure>", resize_page)

        def on_mousewheel(event):
            # Windows / macOS / X11-compatible best effort.
            if getattr(event, "delta", 0):
                canvas.yview_scroll(int(-event.delta / 120), "units")
            elif getattr(event, "num", None) == 4:
                canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(3, "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel, add="+")
        canvas.bind_all("<Button-4>", on_mousewheel, add="+")
        canvas.bind_all("<Button-5>", on_mousewheel, add="+")

        header = ttk.Frame(page)
        header.pack(fill="x", pady=(0, 12))
        title_box = ttk.Frame(header)
        title_box.pack(side="left", fill="x", expand=True)
        ttk.Label(title_box, text=APP_NAME, style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_box, text=APP_SUBTITLE, style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))
        header_right = ttk.Frame(header)
        header_right.pack(side="right")
        self.header_badge = ttk.Label(header_right, text="● READY", style="Status.TLabel")
        self.header_badge.pack(anchor="e")

        workspace = ttk.Frame(page)
        workspace.pack(fill="x", expand=False)
        workspace.columnconfigure(0, weight=0)
        workspace.columnconfigure(1, weight=1)
        self.build_sidebar(workspace)
        self.build_main(workspace)

        footer = ttk.Frame(page)
        footer.pack(fill="x", pady=(10, 0))
        self.footer_capabilities = ttk.Label(footer, text="Checking environment…", style="Muted.TLabel")
        self.footer_capabilities.pack(side="left")
        ttk.Label(footer, text="Scroll the page to reach the full workspace", style="Muted.TLabel").pack(side="right")

    def build_sidebar(self, parent):
        sidebar = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        sidebar.configure(width=275)
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="WORKFLOW", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.input_summary = self.add_sidebar_card(sidebar, 1, "1", "Input", "Choose audio/video")
        self.model_summary = self.add_sidebar_card(sidebar, 2, "2", "Model", "Select GGUF model")
        self.output_summary = self.add_sidebar_card(sidebar, 3, "3", "Output", "Choose file + format")

        ttk.Separator(sidebar).grid(row=4, column=0, sticky="ew", pady=14)
        ttk.Label(sidebar, text="RUN STATUS", style="Section.TLabel").grid(row=5, column=0, sticky="w", pady=(0, 10))

        status_card = ttk.Frame(sidebar, style="Card.TFrame", padding=12)
        status_card.grid(row=6, column=0, sticky="ew")
        self.sidebar_status = ttk.Label(status_card, text="Ready", style="Card.TLabel", font=("Segoe UI", 10, "bold"))
        self.sidebar_status.pack(anchor="w")
        self.sidebar_elapsed = ttk.Label(status_card, text="Elapsed 00:00", style="MetricCaption.TLabel")
        self.sidebar_elapsed.pack(anchor="w", pady=(4, 0))

        self.start_btn = ttk.Button(sidebar, text="Start Transcription", style="Accent.TButton", command=self.start_transcription)
        self.start_btn.grid(row=7, column=0, sticky="ew", pady=(18, 7))
        self.stop_btn = ttk.Button(sidebar, text="Stop", style="Danger.TButton", command=self.cancel_transcription, state="disabled")
        self.stop_btn.grid(row=8, column=0, sticky="ew")

        ttk.Separator(sidebar).grid(row=9, column=0, sticky="ew", pady=14)
        self.dnd_hint = ttk.Label(
            sidebar,
            text=("Drag & drop is enabled." if HAVE_DND else "Drag & drop optional: install tkinterdnd2."),
            style="Muted.TLabel",
            wraplength=235,
        )
        self.dnd_hint.grid(row=8, column=0, sticky="w")

    def add_sidebar_card(self, parent, row, number, title, caption):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=10)
        frame.grid(row=row, column=0, sticky="ew", pady=4)
        num = tk.Label(frame, text=number, bg=self.ACCENT, fg="#08111f", font=("Segoe UI", 9, "bold"), width=2)
        num.pack(side="left", padx=(0, 9))
        text = ttk.Frame(frame, style="Card.TFrame")
        text.pack(side="left", fill="x", expand=True)
        ttk.Label(text, text=title, style="Card.TLabel", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        label = ttk.Label(text, text=caption, style="MetricCaption.TLabel", wraplength=180)
        label.pack(anchor="w", pady=(2, 0))
        return label

    def build_main(self, parent):
        main = ttk.Frame(parent)
        main.grid(row=0, column=1, sticky="new")
        main.columnconfigure(0, weight=1)

        metrics = ttk.Frame(main)
        metrics.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for i in range(4):
            metrics.columnconfigure(i, weight=1)
        self.metric_file = self.make_metric_card(metrics, 0, "INPUT", "No file")
        self.metric_model = self.make_metric_card(metrics, 1, "MODEL", "Not selected")
        self.metric_mode = self.make_metric_card(metrics, 2, "MODE", "Multi-speaker")
        self.metric_duration = self.make_metric_card(metrics, 3, "DURATION", "—")

        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=1, column=0, sticky="ew")

        input_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=16)
        runtime_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=16)
        output_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=16)

        self.notebook.add(input_tab, text="  Input  ")
        self.notebook.add(runtime_tab, text="  Engine  ")
        self.notebook.add(output_tab, text="  Output  ")

        # New MOSS Chunk tab
        moss_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=16)
        self.notebook.add(moss_tab, text=" MOSS Chunk ")
        self.moss_tab = MossTab(moss_tab, self)

        for tab in (input_tab, runtime_tab, output_tab):
            tab.columnconfigure(1, weight=1)

        self.populate_input_tab(input_tab)
        self.populate_runtime_tab(runtime_tab)
        self.populate_output_tab(output_tab)

        # Deliberately stacked rather than a collapsible/resize-dependent pane.
        # The entire page is vertically scrollable, so both views have a
        # predictable usable height on small screens.
        content = ttk.Frame(main)
        content.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        content.columnconfigure(0, weight=1)

        transcript_frame = ttk.Frame(content, style="Panel.TFrame", padding=10, height=390)
        transcript_frame.grid(row=0, column=0, sticky="ew")
        transcript_frame.grid_propagate(False)

        separator = ttk.Separator(content, orient="horizontal")
        separator.grid(row=1, column=0, sticky="ew", pady=10)

        log_frame = ttk.Frame(content, style="Panel.TFrame", padding=10, height=430)
        log_frame.grid(row=2, column=0, sticky="ew")
        log_frame.grid_propagate(False)

        self.build_transcript_panel(transcript_frame)
        self.build_log_panel(log_frame)

    def make_metric_card(self, parent, col, caption, value):
        card = ttk.Frame(parent, style="Card.TFrame", padding=11)
        card.grid(row=0, column=col, sticky="nsew", padx=3)
        ttk.Label(card, text=caption, style="MetricCaption.TLabel").pack(anchor="w")
        value_label = ttk.Label(card, text=value, style="Metric.TLabel")
        value_label.pack(anchor="w", pady=(4, 0))
        return value_label

    def populate_input_tab(self, tab):
        ttk.Label(tab, text="Audio / video source", style="Panel.TLabel", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self.audio_entry = ttk.Entry(tab, textvariable=self.audio_path_var)
        self.audio_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 8))
        ttk.Button(tab, text="Browse…", command=self.browse_audio).grid(row=1, column=2, sticky="ew")
        ttk.Label(tab, text="Supported: WAV, MP3, FLAC, OGG, M4A, WMA, AAC, MP4, WEBM", style="Muted.TLabel").grid(row=2, column=0, columnspan=3, sticky="w", pady=(5, 16))

        speaker = ttk.Frame(tab, style="Card.TFrame", padding=12)
        speaker.grid(row=3, column=0, columnspan=3, sticky="ew")
        ttk.Label(speaker, text="Speaker attribution", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Radiobutton(speaker, text="Multi-speaker / diarization", value="multi", variable=self.speaker_mode_var, command=self.refresh_summaries).pack(anchor="w", pady=(8, 3))
        ttk.Radiobutton(speaker, text="Single-speaker / no diarization", value="single", variable=self.speaker_mode_var, command=self.refresh_summaries).pack(anchor="w")

    def populate_runtime_tab(self, tab):
        ttk.Label(tab, text="GGUF model", style="Panel.TLabel", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 10))
        self.model_entry = ttk.Entry(tab, textvariable=self.model_path_var)
        self.model_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 6))
        ttk.Button(tab, text="Browse…", command=self.browse_model).grid(row=1, column=2, padx=3)
        ttk.Button(tab, text="Detect", command=self.detect_default_model).grid(row=1, column=3, padx=(3, 0))
        ttk.Button(tab, text="Diagnose", command=self.diagnose_model).grid(row=1, column=4, padx=(6, 0))
        self.model_status_label = ttk.Label(tab, text="Model compatibility: not checked", style="Muted.TLabel", wraplength=700)
        self.model_status_label.grid(row=2, column=0, columnspan=4, sticky="w", pady=(5, 0))

        ttk.Label(tab, text="Transcribe executable").grid(row=2, column=0, sticky="w", pady=(15, 6))
        self.binary_entry = ttk.Entry(tab, textvariable=self.binary_path_var)
        self.binary_entry.grid(row=3, column=0, columnspan=2, sticky="ew", padx=(0, 6))
        ttk.Button(tab, text="Browse…", command=self.browse_binary).grid(row=3, column=2, padx=3)

        ttk.Label(tab, text="Compute backend").grid(row=4, column=0, sticky="w", pady=(15, 6))
        self.backend_combo = ttk.Combobox(tab, textvariable=self.backend_var, values=["Vulkan", "Auto", "CPU"], state="readonly", width=20)
        self.backend_combo.grid(row=5, column=0, sticky="w")
        self.backend_combo.bind("<<ComboboxSelected>>", lambda _e: self.update_backend_ui())
        self.backend_status_label = ttk.Label(tab, text="Vulkan is required; the executable must be built with TRANSCRIBE_VULKAN=ON.", style="Muted.TLabel", wraplength=650)
        self.backend_status_label.grid(row=5, column=1, columnspan=3, sticky="w", padx=(10, 0))

        ttk.Label(tab, text="Vulkan device IDs").grid(row=6, column=0, sticky="w", pady=(15, 6))
        device_row = ttk.Frame(tab, style="Panel.TFrame")
        device_row.grid(row=7, column=0, columnspan=4, sticky="ew")
        ttk.Entry(device_row, textvariable=self.vulkan_devices_var, width=18).pack(side="left")
        ttk.Label(device_row, text="Use 0,1 to expose both iGPU + dGPU; 0 or 1 selects one. Applied as GGML_VK_VISIBLE_DEVICES.", style="Muted.TLabel", wraplength=600).pack(side="left", padx=(10, 0))

        ttk.Label(tab, text="CPU threads").grid(row=8, column=0, sticky="w", pady=(15, 6))
        scale = ttk.Scale(tab, from_=1, to=max(1, os.cpu_count() or 16), variable=self.threads_var, orient="horizontal", command=lambda _v: self.on_threads_changed())
        scale.grid(row=9, column=0, columnspan=2, sticky="ew", padx=(0, 8))
        self.threads_label = ttk.Label(tab, text=str(self.resolved_threads()))
        self.threads_label.grid(row=9, column=2, sticky="w")

        ttk.Label(tab, text="Language").grid(row=10, column=0, sticky="w", pady=(15, 6))
        ttk.Combobox(tab, textvariable=self.language_var, values=["Auto", "English (en)", "Chinese (zh)", "Japanese (ja)", "French (fr)", "German (de)"], state="normal", width=20).grid(row=11, column=0, sticky="w")
        ttk.Label(tab, text="Enter an ISO 639-1/639-3 code when it is not listed.", style="Muted.TLabel").grid(row=11, column=1, columnspan=3, sticky="w", padx=(8, 0))

        ttk.Label(tab, text="Timestamps").grid(row=12, column=0, sticky="w", pady=(15, 6))
        ttk.Combobox(tab, textvariable=self.timestamp_var, values=["Auto", "Segment", "None"], state="readonly", width=20).grid(row=13, column=0, sticky="w")

        ttk.Label(tab, text="Backend arguments are generated safely as an argument list; paths are not shell-interpolated.", style="Muted.TLabel", wraplength=650).grid(row=14, column=0, columnspan=4, sticky="w", pady=(18, 0))

    def populate_output_tab(self, tab):
        ttk.Label(tab, text="Transcript destination", style="Panel.TLabel", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self.output_entry = ttk.Entry(tab, textvariable=self.output_path_var)
        self.output_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 8))
        ttk.Button(tab, text="Save as…", command=self.browse_output).grid(row=1, column=2)

        ttk.Label(tab, text="Format").grid(row=2, column=0, sticky="w", pady=(15, 6))
        fmt = ttk.Combobox(tab, textvariable=self.output_format_var, values=["TXT", "MARKDOWN", "SRT", "JSON", "RAW LOG"], state="readonly", width=16)
        fmt.grid(row=3, column=0, sticky="w")
        fmt.bind("<<ComboboxSelected>>", lambda _e: self.update_output_extension())
        ttk.Label(tab, text="TXT clean text · MARKDOWN readable report · SRT subtitles · JSON structured segments · RAW LOG exact process output", style="Muted.TLabel").grid(row=3, column=1, columnspan=3, sticky="w", padx=(8, 0))

        ttk.Label(tab, text="Output is written only after the process exits successfully.", style="Muted.TLabel").grid(row=4, column=0, columnspan=3, sticky="w", pady=(18, 0))

    def build_transcript_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        top = ttk.Frame(parent, style="Panel.TFrame")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(top, text="TRANSCRIPT", style="Section.TLabel").pack(side="left")
        ttk.Checkbutton(top, text="Auto-scroll", variable=self.autoscroll_var).pack(side="right")
        self.transcript_count = ttk.Label(top, text="0 lines", style="MetricCaption.TLabel")
        self.transcript_count.pack(side="right", padx=(10, 18))
        self.transcript_text = tk.Text(parent, wrap="none", state="disabled", height=14, undo=False)
        self.configure_text_widget(self.transcript_text)
        self.transcript_text.grid(row=1, column=0, sticky="nsew")
        ysb = ttk.Scrollbar(parent, orient="vertical", command=self.transcript_text.yview)
        ysb.grid(row=1, column=1, sticky="ns")
        xsb = ttk.Scrollbar(parent, orient="horizontal", command=self.transcript_text.xview)
        xsb.grid(row=2, column=0, sticky="ew")
        self.transcript_text.config(yscrollcommand=ysb.set, xscrollcommand=xsb.set)

    def build_log_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        top = ttk.Frame(parent, style="Panel.TFrame")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(top, text="PROCESS LOG", style="Section.TLabel").pack(side="left")
        ttk.Button(top, text="Clear", command=self.clear_log).pack(side="right")
        ttk.Button(top, text="Copy", command=self.copy_log).pack(side="right", padx=6)
        ttk.Button(top, text="Save…", command=self.save_log).pack(side="right")
        self.log_count = ttk.Label(top, text="0 lines", style="MetricCaption.TLabel")
        self.log_count.pack(side="right", padx=(10, 18))
        self.log_text = tk.Text(parent, wrap="none", state="disabled", height=8, undo=False)
        self.configure_text_widget(self.log_text)
        self.log_text.grid(row=1, column=0, sticky="nsew")
        ysb = ttk.Scrollbar(parent, orient="vertical", command=self.log_text.yview)
        ysb.grid(row=1, column=1, sticky="ns")
        xsb = ttk.Scrollbar(parent, orient="horizontal", command=self.log_text.xview)
        xsb.grid(row=2, column=0, sticky="ew")
        self.log_text.config(yscrollcommand=ysb.set, xscrollcommand=xsb.set)

        status_row = ttk.Frame(parent, style="Panel.TFrame")
        status_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        status_row.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(status_row, variable=self.progress_var, maximum=100, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(status_row, textvariable=self.elapsed_var, style="Status.TLabel", width=8).grid(row=0, column=1)
        ttk.Label(status_row, textvariable=self.status_var, style="Status.TLabel", width=24).grid(row=0, column=2, padx=(6, 0))

        ttk.Label(parent, text="COMMAND", style="MetricCaption.TLabel").grid(row=4, column=0, sticky="w", pady=(8, 2))
        self.command_text = ttk.Entry(parent, state="readonly")
        self.command_text.grid(row=5, column=0, columnspan=2, sticky="ew")

        if HAVE_DND:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self.on_drop)

    def browse_model(self):
        filename = filedialog.askopenfilename(title="Select GGUF Model", filetypes=[("Model Files", "*.gguf *.bin"), ("All Files", "*.*")])
        if filename:
            self.model_path_var.set(filename)
            self.save_settings()
            self.refresh_summaries()
            self.update_model_status()

    def browse_binary(self):
        filename = filedialog.askopenfilename(title="Select Transcribe Executable", filetypes=[("Executable Files", "*.exe"), ("All Files", "*.*")])
        if filename:
            self.binary_path_var.set(filename)
            self.save_settings()
            self.update_capabilities()
        self.update_backend_ui()

    def browse_audio(self):
        filename = filedialog.askopenfilename(title="Select Audio / Video File", filetypes=SUPPORTED_AUDIO)
        if filename:
            self.set_audio(filename)

    def browse_output(self):
        ext = self.current_extension()
        filename = filedialog.asksaveasfilename(title="Choose transcript output", defaultextension=ext, filetypes=[(f"{self.output_format_var.get()} output", f"*{ext}"), ("All Files", "*.*")])
        if filename:
            self.output_path_var.set(filename)
            self.save_settings()
            self.refresh_summaries()

    def set_audio(self, filename):
        self.audio_path_var.set(filename)
        self.set_default_output()
        self.current_audio_duration = None
        self.save_settings()
        self.refresh_summaries()
        threading.Thread(target=self.probe_duration_async, args=(filename,), daemon=True).start()

    def probe_duration_async(self, filename):
        duration = self.get_audio_duration(filename)
        self.ui_queue.put(("duration", (filename, duration)))

    def on_drop(self, event):
        try:
            paths = self.root.tk.splitlist(event.data)
        except tk.TclError:
            paths = [event.data]
        if not paths:
            return
        for path in paths:
            if path.lower().endswith((".gguf", ".bin")):
                self.model_path_var.set(path)
            else:
                self.set_audio(path)
                break
        self.save_settings()
        self.refresh_summaries()

    # update_model_status() and diagnose_model() live in ModelCompatibilityMixin
    # (app/models/compatibility.py). They used to be duplicated here too; since
    # AppWindowMixin comes before ModelCompatibilityMixin in the mixin order in
    # main.py, this copy was silently winning and the compatibility.py methods
    # were dead code. Removed rather than fixed in place, since compatibility
    # logic belongs in the compatibility module, not the GUI shell.

    def update_backend_ui(self):
        selected = self.backend_var.get()
        if hasattr(self, "backend_status_label"):
            if selected == "Vulkan":
                self.backend_status_label.config(text="Vulkan REQUIRED: the selected transcribe-cli must be a Vulkan build; CPU-only binaries will be rejected.")
            elif selected == "Auto":
                self.backend_status_label.config(text="Auto: use the backend selected by the executable; this may fall back to CPU.")
            else:
                self.backend_status_label.config(text="CPU: GPU acceleration is disabled for this run.")

    def resolved_threads(self):
        try:
            value = int(round(float(self.threads_var.get())))
        except (TypeError, ValueError):
            value = 1
        return max(1, value)

    def on_threads_changed(self):
        if hasattr(self, "threads_label"):
            self.threads_label.config(text=str(self.resolved_threads()))

    def current_extension(self):
        return {"TXT": ".txt", "MARKDOWN": ".md", "SRT": ".srt", "JSON": ".json", "RAW LOG": ".log"}.get(self.output_format_var.get(), ".txt")

    def update_output_extension(self):
        current = self.output_path_var.get().strip()
        if not current:
            self.set_default_output()
        else:
            self.output_path_var.set(str(Path(current).with_suffix(self.current_extension())))
        self.save_settings()
        self.refresh_summaries()

    def set_default_output(self):
        audio = self.audio_path_var.get().strip()
        if audio:
            self.output_path_var.set(str(Path(audio).with_suffix(self.current_extension())))

    def append_log(self, text):
        if not text:
            return
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, text)
        if self.autoscroll_var.get():
            self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        if hasattr(self, "log_count"):
            lines = int(float(self.log_text.index("end-1c").split(".")[0]))
            self.log_count.config(text=f"{lines:,} lines")

    def set_transcript_text(self, text):
        self.transcript_text.config(state="normal")
        self.transcript_text.delete("1.0", tk.END)
        self.transcript_text.insert("1.0", text or "")
        if self.autoscroll_var.get():
            self.transcript_text.see(tk.END)
        self.transcript_text.config(state="disabled")
        if hasattr(self, "transcript_count"):
            lines = int(float(self.transcript_text.index("end-1c").split(".")[0]))
            self.transcript_count.config(text=f"{lines:,} lines")

    def clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")
        if hasattr(self, "log_count"):
            self.log_count.config(text="0 lines")

    def copy_log(self):
        text = self.log_text.get("1.0", tk.END).rstrip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Log copied")

    def save_log(self):
        filename = filedialog.asksaveasfilename(title="Save process log", defaultextension=".log", filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")])
        if not filename:
            return
        try:
            Path(filename).write_text(self.log_text.get("1.0", tk.END), encoding="utf-8")
            self.status_var.set("Log saved")
        except OSError as exc:
            messagebox.showerror("Save Error", str(exc))

    def refresh_summaries(self):
        audio = self.audio_path_var.get().strip()
        model = self.model_path_var.get().strip()
        output = self.output_path_var.get().strip()
        mode = "Multi-speaker" if self.speaker_mode_var.get() == "multi" else "Single-speaker"
        file_name = Path(audio).name if audio else "No file"
        model_name = Path(model).name if model else "Not selected"
        duration = self.current_audio_duration
        duration_text = self.format_seconds(duration) if duration else "—"

        if hasattr(self, "metric_file"):
            self.metric_file.config(text=file_name[:28] + ("…" if len(file_name) > 28 else ""))
            self.metric_model.config(text=model_name[:28] + ("…" if len(model_name) > 28 else ""))
            self.metric_mode.config(text=mode)
            self.metric_duration.config(text=duration_text)
        if hasattr(self, "input_summary"):
            self.input_summary.config(text=file_name[:32] + ("…" if len(file_name) > 32 else ""))
            self.model_summary.config(text=model_name[:32] + ("…" if len(model_name) > 32 else ""))
            self.output_summary.config(text=(Path(output).name if output else "No destination")[:32])

    def update_capabilities(self):
        ffmpeg = bool(shutil.which("ffmpeg"))
        ffprobe = bool(shutil.which("ffprobe"))
        binary = self.binary_path_var.get().strip()
        binary_ok = bool(shutil.which(binary) or Path(binary).is_file()) if binary else False
        dnd = HAVE_DND
        bits = [
            f"Transcriber: {'OK' if binary_ok else 'not found'}",
            f"ffmpeg: {'OK' if ffmpeg else 'missing'}",
            f"ffprobe: {'OK' if ffprobe else 'missing'}",
            f"Drag-drop: {'ON' if dnd else 'OFF'}",
        ]
        if hasattr(self, "footer_capabilities"):
            self.footer_capabilities.config(text="  ·  ".join(bits))

    def update_elapsed(self):
        if self.job_active and self.started_at:
            seconds = int(time.monotonic() - self.started_at)
            formatted = self.format_seconds(seconds, always_hours=True)
            self.elapsed_var.set(formatted)
            if hasattr(self, "sidebar_elapsed"):
                self.sidebar_elapsed.config(text=f"Elapsed {formatted}")
        self.root.after(500, self.update_elapsed)

    def poll_ui_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    self.append_log(payload)
                elif kind == "transcript":
                    self.set_transcript_text(payload)
                elif kind == "progress":
                    self.progress_var.set(payload)
                elif kind == "indeterminate":
                    self.progress.configure(mode="indeterminate")
                    self.progress.start(12)
                elif kind == "determinate":
                    self.progress.stop()
                    self.progress.configure(mode="determinate")
                elif kind == "status":
                    self.status_var.set(payload)
                    self.sidebar_status.config(text=payload)
                    self.header_badge.config(text=f"● {str(payload).upper()[:18]}")
                elif kind == "duration":
                    filename, duration = payload
                    if filename == self.audio_path_var.get().strip():
                        self.current_audio_duration = duration
                        self.refresh_summaries()
                elif kind == "command":
                    self.set_command_display(payload)
                elif kind == "backend_help":
                    self.backend_help = payload
                elif kind == "backend_runtime":
                    self.current_backend = payload
                    if hasattr(self, "backend_status_label"):
                        self.backend_status_label.config(text=f"Runtime backend: {payload}")
                elif kind == "segments":
                    self.update_live_transcript(payload)
                elif kind == "finished":
                    self.finish_job(*payload)
        except queue.Empty:
            pass
        self.root.after(75, self.poll_ui_queue)

    def set_command_display(self, command):
        self.command_text.config(state="normal")
        self.command_text.delete(0, tk.END)
        self.command_text.insert(0, command)
        self.command_text.config(state="readonly")

    def update_live_transcript(self, segments):
        if not segments:
            return
        transcript = self.extract_transcript("", segments)
        self.ui_queue.put(("transcript", transcript))

    def on_close(self):
        if self.job_active:
            if not messagebox.askyesno("Transcription Running", "A transcription is still running. Stop it and close the application?"):
                return
            self.cancel_requested.set()
            self.terminate_process()
        self.save_settings()
        self.root.destroy()