import os
import re
import subprocess
from pathlib import Path

class VulkanBackendMixin:
    def update_backend_ui(self):
        selected = self.backend_var.get()
        if hasattr(self, "backend_status_label"):
            if selected == "Vulkan":
                self.backend_status_label.config(text="Vulkan REQUIRED: the selected transcribe-cli must be a Vulkan build; CPU-only binaries will be rejected.")
            elif selected == "Auto":
                self.backend_status_label.config(text="Auto: use the backend selected by the executable; this may fall back to CPU.")
            else:
                self.backend_status_label.config(text="CPU: GPU acceleration is disabled for this run.")
    def probe_backend_help(self, binary):
        try:
            result = subprocess.run(
                [binary, "--help"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", timeout=8,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.backend_help = result.stdout or ""
            self.backend_options = set(re.findall(r"(?:^|\s)(--[a-zA-Z0-9][a-zA-Z0-9_-]*)", self.backend_help))
            self.ui_queue.put(("backend_help", self.backend_help))
            return self.backend_help
        except (OSError, subprocess.SubprocessError) as exc:
            self.backend_help = f"Unable to query backend help: {exc}"
            self.backend_options = set()
            return self.backend_help
    def backend_supports(self, *options):
        if not self.backend_options:
            return True
        return any(option in self.backend_options for option in options)
    def verify_requested_backend(self, binary):
        requested = self.backend_var.get().strip().lower()
        if requested == "cpu":
            return True, "CPU backend explicitly selected."
        if requested == "auto":
            return True, "Auto backend selected; executable chooses the available backend."
        try:
            result = subprocess.run(
                [binary, "--backend", "vulkan", "--help"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", timeout=8,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            text = result.stdout or ""
            lowered = text.lower()
            if result.returncode != 0 and any(x in lowered for x in ("unknown option", "unrecognized option", "invalid value", "invalid argument")):
                return False, "The executable rejected '--backend vulkan'. It is not a Vulkan-capable build."
            return True, "Vulkan backend requested explicitly."
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"Unable to verify Vulkan backend support: {exc}"
    def vulkan_environment(self):
        """Return a subprocess environment with the requested Vulkan devices visible."""
        env = os.environ.copy()
        devices = self.vulkan_devices_var.get().strip()
        if self.backend_var.get().strip().lower() == "vulkan" and devices:
            if not re.fullmatch(r"\d+(?:\s*,\s*\d+)*", devices):
                raise ValueError("Vulkan device IDs must look like 0, 1, or 0,1.")
            normalized = ",".join(part.strip() for part in devices.split(","))
            env["GGML_VK_VISIBLE_DEVICES"] = normalized
        elif "GGML_VK_VISIBLE_DEVICES" in env:
            env.pop("GGML_VK_VISIBLE_DEVICES", None)
        return env

