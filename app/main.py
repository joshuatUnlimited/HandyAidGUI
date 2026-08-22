import os
import queue
import re
import json
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from app.config.settings import APP_NAME, APP_SUBTITLE, SettingsMixin

# Inject values into app_window module
from app.gui import app_window as _app_window
_app_window.APP_NAME = APP_NAME
_app_window.APP_SUBTITLE = APP_SUBTITLE

from app.gui.app_window import AppWindowMixin, HAVE_DND, TkinterDnD
from app.gui.theme import ThemeMixin
from app.models.gguf import GGUFModelMixin
from app.models.compatibility import ModelCompatibilityMixin
from app.backends.vulkan import VulkanBackendMixin
from app.engine.command_builder import CommandBuilderMixin
from app.engine.transcriber import TranscriptionEngineMixin
from app.audio.converter import AudioConverterMixin
from app.audio.duration import AudioDurationMixin
from app.parsing.segments import ParsingMixin
from app.output.writers import OutputWriterMixin

# GPU panel import
from app.gui.gpu_panel import GPUControlPanelMixin


# ======================================================================
# REAL GPU MANAGER – uses the transcribe binary to detect Vulkan devices
# ======================================================================
class RealGPUManager:
    def __init__(self, binary_path):
        self.binary_path = binary_path
        self._devices = []          # cache after scan
        self._last_error = None

    def _run_binary_with_flag(self, flag):
        """Run transcribe with the given flag and return stdout + stderr."""
        try:
            result = subprocess.run(
                [self.binary_path, flag],
                capture_output=True,
                text=True,
                timeout=10,
                check=False
            )
            return result.stdout + "\n" + result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            return f"ERROR: {e}"

    def _try_parse_json(self, output):
        """Attempt to parse JSON output from the binary."""
        try:
            data = json.loads(output)
            # Assume data is a list of devices, or a dict with a 'devices' key.
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'devices' in data:
                return data['devices']
            elif isinstance(data, dict) and 'gpus' in data:
                return data['gpus']
        except json.JSONDecodeError:
            pass
        return None

    def _try_parse_text(self, output):
        """
        Heuristic text parsing:
        Look for lines with GPU names and memory.
        Common patterns:
          - "GPU 0: NVIDIA GeForce RTX 3080 (10240 MB)"
          - "Device 0: NVIDIA RTX A6000, VRAM: 49152 MB"
          - "Vulkan Device: ... Memory: ..."
        """
        devices = []
        lines = output.splitlines()
        for line in lines:
            # Try to match: number, colon, name, possibly parentheses with MB.
            # e.g. "GPU 0: NVIDIA GeForce RTX 3080 (10240 MB)"
            match = re.search(r'(?:GPU|Device)\s*(\d+)\s*:\s*([^,\(]+)(?:\s*\((\d+)\s*MB\))?', line, re.IGNORECASE)
            if match:
                idx = int(match.group(1))
                name = match.group(2).strip()
                mem_str = match.group(3)
                total_mem = int(mem_str) if mem_str else 0
                devices.append({
                    'index': idx,
                    'name': name,
                    'total_memory_mb': total_mem,
                    'free_memory_mb': total_mem,   # unknown, assume full free
                    'used_memory_mb': 0,
                })
                continue
            # Another pattern: "Name: ... Memory: ..."
            match = re.search(r'Name:\s*([^\n]+)\s*Memory:\s*(\d+)\s*MB', line, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                total_mem = int(match.group(2))
                devices.append({
                    'index': len(devices),   # sequential index
                    'name': name,
                    'total_memory_mb': total_mem,
                    'free_memory_mb': total_mem,
                    'used_memory_mb': 0,
                })
        return devices

    def scan(self):
        """Return a list of GPU device dicts."""
        # Try a few known flags
        flags_to_try = [
            "--list-gpu",
            "--vulkan-devices",
            "--gpu-info",
            "--print-devices",
            "--list-devices"
        ]
        for flag in flags_to_try:
            output = self._run_binary_with_flag(flag)
            # Skip if output is an error
            if output.startswith("ERROR:"):
                continue
            # Try JSON first
            devices = self._try_parse_json(output)
            if devices is not None:
                self._devices = devices
                self._last_error = None
                return devices
            # Try text parsing
            devices = self._try_parse_text(output)
            if devices:
                self._devices = devices
                self._last_error = None
                return devices

        # If we get here, no flag worked
        self._last_error = "No GPU list flag found; binary may not support listing."
        self._devices = []
        return []

    def enumerate_all(self, extra_arg=None):
        """Same as scan; some versions of the mixin pass an extra argument."""
        return self.scan()

    def get_usage(self, device_index):
        """Return usage info; not easily available, so return None."""
        return None

    def get_memory_info(self, device_index):
        """Return memory info; if we have total, return it."""
        for dev in self._devices:
            if dev.get('index') == device_index:
                return {
                    'total': dev.get('total_memory_mb', 0),
                    'free': dev.get('free_memory_mb', 0),
                    'used': dev.get('used_memory_mb', 0),
                }
        return None

    @property
    def last_error(self):
        return self._last_error
# ======================================================================


class MossTranscribeGUI(
    SettingsMixin,
    ThemeMixin,
    AppWindowMixin,
    GGUFModelMixin,
    ModelCompatibilityMixin,
    VulkanBackendMixin,
    CommandBuilderMixin,
    TranscriptionEngineMixin,
    AudioConverterMixin,
    AudioDurationMixin,
    ParsingMixin,
    OutputWriterMixin,
    GPUControlPanelMixin,          # <-- ADDED
):
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1180x820")
        self.root.minsize(820, 620)
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.process = None
        self.processes = {}
        self.worker = None
        self.cancel_requested = threading.Event()
        self.ui_queue = queue.Queue()
        self.started_at = None
        self.current_audio_duration = None
        self.current_output_path = None
        self.current_audio_path = None
        self.last_segments = []
        self.last_command = []
        self.last_raw_output = ""
        self.job_active = False
        self.transcript_dirty = False
        self._last_ui_update = 0.0
        self.backend_help = ""
        self.backend_options = set()
        self.current_backend = ""
        self.last_failure_details = ""
        self.model_info = {}
        self.compatible_model_candidates = []

        cpu_count = max(1, os.cpu_count() or 4)

        self.model_path_var = tk.StringVar()
        self.binary_path_var = tk.StringVar(
            value="transcribe.exe" if os.name == "nt" else "transcribe"
        )
        self.audio_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.output_format_var = tk.StringVar(value="TXT")
        self.threads_var = tk.IntVar(value=min(4, cpu_count))
        self.backend_var = tk.StringVar(value="Vulkan")
        self.vulkan_devices_var = tk.StringVar(value="0,1")
        self.speaker_mode_var = tk.StringVar(value="multi")
        self.language_var = tk.StringVar(value="Auto")
        self.timestamp_var = tk.StringVar(value="Auto")
        self.status_var = tk.StringVar(value="Ready")
        self.elapsed_var = tk.StringVar(value="00:00")
        self.progress_var = tk.DoubleVar(value=0)
        self.autoscroll_var = tk.BooleanVar(value=True)
        self.theme_var = tk.StringVar(value="dark")

        self.load_settings()
        self.setup_styles()
        self.build_ui()  # creates the notebook (self.notebook)
        self.detect_default_model()
        self.update_output_extension()

        # ========== INITIALISE GPU-RELATED ATTRIBUTES ==========
        self.transcribe_binary = self.binary_path_var.get()
        self.gpu_manager = RealGPUManager(self.transcribe_binary)
        # ========================================================

        # ========== ADD GPU TAB ==========
        if hasattr(self, 'notebook'):
            self.gpu_tab = self.build_gpu_tab(self.notebook)
            self.notebook.add(self.gpu_tab, text="GPU")
        # =================================

        if (
            self.audio_path_var.get().strip()
            and Path(self.audio_path_var.get().strip()).is_file()
        ):
            self.current_audio_duration = None
            threading.Thread(
                target=self.probe_duration_async,
                args=(self.audio_path_var.get().strip(),),
                daemon=True,
            ).start()

        self.poll_ui_queue()
        self.update_elapsed()
        self.update_capabilities()
        self.update_backend_ui()

    # ========== DELEGATE TKINTER METHODS TO self.root ==========
    def after(self, ms, func):
        return self.root.after(ms, func)

    def after_cancel(self, id):
        self.root.after_cancel(id)

    def update_idletasks(self):
        self.root.update_idletasks()
    # ===========================================================


def main():
    if HAVE_DND:
        class SafeTkinterDnD(TkinterDnD.Tk):
            def readprofile(self, _base_name, _class_name):
                return

        root = SafeTkinterDnD()
    else:
        class SafeTk(tk.Tk):
            def readprofile(self, _base_name, _class_name):
                return

        root = SafeTk()

    MossTranscribeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()