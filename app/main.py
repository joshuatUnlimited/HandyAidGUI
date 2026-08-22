import os
import queue
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
from app.backends.gpu_manager import VulkanGPUManager, VulkanGPU
from app.engine.command_builder import CommandBuilderMixin
from app.engine.transcriber import TranscriptionEngineMixin
from app.audio.converter import AudioConverterMixin
from app.audio.duration import AudioDurationMixin
from app.parsing.segments import ParsingMixin
from app.output.writers import OutputWriterMixin

# GPU panel import
from app.gui.gpu_panel import GPUControlPanelMixin


# ======================================================================
# PROPER GPU MANAGER – uses the official --list-devices flag
# ======================================================================
class GPUManager:
    """
    Wraps VulkanGPUManager to provide the interface GPUControlPanelMixin expects.
    """
    def __init__(self, binary_path):
        self.binary_path = binary_path
        self._last_error = None

    def enumerate_all(self, binary: str) -> list:
        """
        Enumerate all Vulkan-capable GPUs using transcribe-cli --list-devices.
        Returns a list of GPUInfo objects (dataclass expected by GPUControlPanelMixin).
        """
        try:
            # Run transcribe --list-devices
            import subprocess
            result = subprocess.run(
                [binary, "--list-devices"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
            )
            text = result.stdout or ""

            if result.returncode != 0:
                self._last_error = f"transcribe --list-devices failed: {text[:500]}"
                return []

            # Parse using the official parser
            vulkan_gpus = VulkanGPUManager.parse_list_devices(text)

            if not vulkan_gpus:
                self._last_error = "No Vulkan-capable GPUs found."
                return []

            # Convert VulkanGPU to GPUInfo (the dataclass GPUControlPanelMixin expects)
            # GPUInfo is defined in gpu_panel.py: index, name, vram_total, vram_free
            from app.gui.gpu_panel import GPUInfo
            return [
                GPUInfo(
                    index=gpu.index,
                    name=gpu.description,
                    vram_total=gpu.memory_total,
                    vram_free=gpu.memory_free,
                )
                for gpu in vulkan_gpus
            ]

        except subprocess.TimeoutExpired:
            self._last_error = "transcribe --list-devices timed out."
            return []
        except FileNotFoundError:
            self._last_error = f"Binary not found: {binary}"
            return []
        except Exception as e:
            self._last_error = f"Error enumerating GPUs: {e}"
            return []

    def probe(self, binary: str, index: int):
        """
        Probe a specific GPU by index. Used for live VRAM refresh.
        """
        try:
            # Use VulkanGPUManager.probe to get fresh data for one GPU
            gpus = VulkanGPUManager.probe(binary, (index,))
            if gpus:
                gpu = gpus[0]
                from app.gui.gpu_panel import GPUInfo
                return GPUInfo(
                    index=gpu.index,
                    name=gpu.description,
                    vram_total=gpu.memory_total,
                    vram_free=gpu.memory_free,
                )
        except Exception:
            pass
        return None

    @property
    def last_error(self):
        return self._last_error


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
        self.gpu_manager = GPUManager(self.transcribe_binary)
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