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
# REAL GPU MANAGER – detects Vulkan devices using transcribe + vulkaninfo
# ======================================================================
class RealGPUManager:
    def __init__(self, binary_path):
        self.binary_path = binary_path
        self._devices = []
        self._last_error = None
        self._debug_logs = []   # store recent debug output

    def _run_command(self, cmd, timeout=10):
        """Run a command and return (stdout, stderr, returncode)."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", f"Command timed out after {timeout}s", -1
        except FileNotFoundError:
            return "", f"Command not found: {cmd[0]}", -1
        except Exception as e:
            return "", str(e), -1

    def _try_flags(self, flags):
        """Try a list of flags on transcribe binary. Return (output, flag_used)."""
        for flag in flags:
            stdout, stderr, rc = self._run_command([self.binary_path, flag])
            combined = stdout + "\n" + stderr
            if rc == 0 and combined.strip():
                self._debug_logs.append(f"Flag {flag} returned output (len={len(combined)})")
                return combined, flag
            else:
                self._debug_logs.append(f"Flag {flag} failed (rc={rc})")
        return None, None

    def _parse_json_devices(self, output):
        """Parse JSON output; expects list or dict with devices/gpus."""
        try:
            data = json.loads(output)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ('devices', 'gpus', 'vulkan_devices', 'gpu_devices'):
                    if key in data and isinstance(data[key], list):
                        return data[key]
        except json.JSONDecodeError:
            pass
        return None

    def _parse_text_devices(self, output):
        """
        Parse common text formats:
          - "GPU 0: NVIDIA GeForce RTX 3080 (10240 MB)"
          - "Device 0: ... Memory: 8192 MB"
          - "Name: ... Memory: ..."
        Also handles Vulkaninfo format:
          "GPU id : 0 (NVIDIA GeForce RTX 3080)"
          "Dedicated video memory: 10240 MB"
        """
        devices = []
        lines = output.splitlines()
        # First try to find structured table with columns
        # Many tools output a table: index, name, memory
        # We'll try to parse lines with numbers and memory.
        # Collect all potential device entries.
        # We'll use regex with multiple patterns.
        patterns = [
            # Pattern 1: "GPU 0: Name (MB)"
            r'(?:GPU|Device)\s*(\d+)\s*:\s*([^,\(]+)(?:\s*\((\d+)\s*MB\))?',
            # Pattern 2: "Name: ..., Memory: 10240 MB"
            r'Name:\s*([^\n]+)\s*Memory:\s*(\d+)\s*MB',
            # Pattern 3: "GPU id : 0 (Name)" then later line with memory
            r'GPU id\s*:\s*(\d+)\s*\(([^)]+)\)',
            # Pattern 4: "Device 0: NVIDIA GeForce RTX 3080, VRAM: 10240 MB"
            r'Device\s*(\d+)\s*:\s*([^,]+),\s*VRAM:\s*(\d+)\s*MB',
        ]
        # We'll do a two-pass: first gather all matches, then merge if needed.
        for pattern in patterns:
            for match in re.finditer(pattern, output, re.IGNORECASE):
                groups = match.groups()
                if len(groups) == 3:
                    idx, name, mem = groups
                    devices.append({
                        'index': int(idx),
                        'name': name.strip(),
                        'total_memory_mb': int(mem) if mem else 0,
                        'free_memory_mb': int(mem) if mem else 0,
                        'used_memory_mb': 0,
                    })
                elif len(groups) == 2:
                    # Could be name & memory, or index & name
                    # Let's see if second group is numeric -> it's memory, else name.
                    if groups[1].isdigit():
                        # name and memory
                        name, mem = groups
                        idx = len(devices)  # assign sequential
                        devices.append({
                            'index': idx,
                            'name': name.strip(),
                            'total_memory_mb': int(mem),
                            'free_memory_mb': int(mem),
                            'used_memory_mb': 0,
                        })
                    else:
                        # likely index and name, but we need memory from elsewhere
                        idx, name = groups
                        # We'll later try to find memory line for this device
                        # For now, store as placeholder
                        pass
        # If we found devices with index, we can try to find memory lines
        # that match "Dedicated video memory: X MB" for each.
        mem_lines = re.findall(r'Dedicated video memory:\s*(\d+)\s*MB', output, re.IGNORECASE)
        for i, mem in enumerate(mem_lines):
            if i < len(devices):
                devices[i]['total_memory_mb'] = int(mem)
                devices[i]['free_memory_mb'] = int(mem)
        return devices

    def _vulkaninfo_fallback(self):
        """Try to use vulkaninfo to get devices."""
        stdout, stderr, rc = self._run_command(['vulkaninfo', '--summary'], timeout=15)
        if rc != 0:
            return []
        # Parse summary for GPUs
        devices = []
        # Look for lines like "GPU id : 0 (NVIDIA GeForce RTX 3080)"
        for line in stdout.splitlines():
            match = re.search(r'GPU id\s*:\s*(\d+)\s*\(([^)]+)\)', line)
            if match:
                idx = int(match.group(1))
                name = match.group(2).strip()
                devices.append({
                    'index': idx,
                    'name': name,
                    'total_memory_mb': 0,  # we'll fill later
                    'free_memory_mb': 0,
                    'used_memory_mb': 0,
                })
        # Try to get memory from lines like "Dedicated video memory: 10240 MB"
        for dev in devices:
            # Find the memory line near the device ID
            # We'll search for "Dedicated video memory: X MB" after the device line
            # Simpler: find all memory lines and assign in order.
        mem_lines = re.findall(r'Dedicated video memory:\s*(\d+)\s*MB', stdout, re.IGNORECASE)
        for i, mem in enumerate(mem_lines):
            if i < len(devices):
                devices[i]['total_memory_mb'] = int(mem)
                devices[i]['free_memory_mb'] = int(mem)
        # If no memory found, try "Device memory: X MB"
        if not mem_lines:
            mem_lines = re.findall(r'Device memory:\s*(\d+)\s*MB', stdout, re.IGNORECASE)
            for i, mem in enumerate(mem_lines):
                if i < len(devices):
                    devices[i]['total_memory_mb'] = int(mem)
                    devices[i]['free_memory_mb'] = int(mem)
        return devices

    def scan(self):
        """Return list of GPU devices."""
        # First check if binary exists
        if not os.path.exists(self.binary_path):
            self._last_error = f"Transcribe binary not found at: {self.binary_path}"
            self._devices = []
            return []

        # Flags to try, ordered by likelihood
        flags_to_try = [
            "--list-vulkan",
            "--list-gpu",
            "--gpu-devices",
            "--show-gpu",
            "--list-devices",
            "--gpu-list",
            "--vulkan-devices",
            "--print-devices",
            "--help"  # often contains flag list; we can parse help to find the right flag
        ]
        output, used_flag = self._try_flags(flags_to_try)

        devices = []
        if output:
            # Try JSON
            parsed = self._parse_json_devices(output)
            if parsed:
                devices = parsed
            else:
                # Try text parsing
                devices = self._parse_text_devices(output)

        # If we have devices, return them
        if devices:
            self._devices = devices
            self._last_error = None
            return devices

        # If transcribe didn't return anything, try vulkaninfo
        self._debug_logs.append("Transcribe flags failed, trying vulkaninfo fallback")
        devices = self._vulkaninfo_fallback()
        if devices:
            self._devices = devices
            self._last_error = None
            return devices

        # Nothing worked
        self._last_error = "No Vulkan-capable GPUs found. Tried transcribe flags and vulkaninfo."
        self._devices = []
        return []

    def enumerate_all(self, extra_arg=None):
        return self.scan()

    def get_usage(self, device_index):
        return None  # Not available

    def get_memory_info(self, device_index):
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

    def get_debug_logs(self):
        return self._debug_logs
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
    GPUControlPanelMixin,
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
        self.build_ui()
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

        # Optional: print debug logs to console on startup (for diagnosis)
        # print("GPU Manager debug logs:", self.gpu_manager.get_debug_logs())

    # ========== DELEGATE TKINTER METHODS ==========
    def after(self, ms, func):
        return self.root.after(ms, func)

    def after_cancel(self, id):
        self.root.after_cancel(id)

    def update_idletasks(self):
        self.root.update_idletasks()
    # ==============================================


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