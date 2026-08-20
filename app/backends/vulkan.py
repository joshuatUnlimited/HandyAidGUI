import os
import re
import subprocess

from app.backends.gpu_manager import VulkanGPUManager


class VulkanBackendMixin:
    def update_backend_ui(self):
        selected = self.backend_var.get()

        if hasattr(self, "backend_status_label"):
            if selected == "Vulkan":
                self.backend_status_label.config(
                    text=(
                        "Vulkan: selected GPUs can be balanced automatically "
                        "for single-speaker transcription. Each worker is "
                        "explicitly pinned to one Vulkan adapter."
                    )
                )
            elif selected == "Auto":
                self.backend_status_label.config(
                    text=(
                        "Auto: use the backend selected by the executable; "
                        "this may fall back to CPU."
                    )
                )
            else:
                self.backend_status_label.config(
                    text=(
                        "CPU: GPU acceleration is disabled for this run."
                    )
                )

    def probe_backend_help(self, binary):
        try:
            result = subprocess.run(
                [binary, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if os.name == "nt"
                    else 0
                ),
            )

            self.backend_help = result.stdout or ""

            self.backend_options = set(
                re.findall(
                    r"(?:^|\s)(--[a-zA-Z0-9]"
                    r"[a-zA-Z0-9_-]*)",
                    self.backend_help,
                )
            )

            self.ui_queue.put(
                ("backend_help", self.backend_help)
            )
            return self.backend_help

        except (OSError, subprocess.SubprocessError) as exc:
            self.backend_help = (
                f"Unable to query backend help: {exc}"
            )
            self.backend_options = set()
            return self.backend_help

    def backend_supports(self, *options):
        if not self.backend_options:
            return True

        return any(
            option in self.backend_options
            for option in options
        )

    def verify_requested_backend(self, binary):
        requested = (
            self.backend_var.get()
            .strip()
            .lower()
        )

        if requested == "cpu":
            return True, "CPU backend explicitly selected."

        if requested == "auto":
            return (
                True,
                "Auto backend selected; executable chooses "
                "the available backend.",
            )

        try:
            result = subprocess.run(
                [
                    binary,
                    "--backend",
                    "vulkan",
                    "--help",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if os.name == "nt"
                    else 0
                ),
            )

            text = result.stdout or ""
            lowered = text.lower()

            if (
                result.returncode != 0
                and any(
                    x in lowered
                    for x in (
                        "unknown option",
                        "unrecognized option",
                        "invalid value",
                        "invalid argument",
                    )
                )
            ):
                return (
                    False,
                    "The executable rejected "
                    "'--backend vulkan'. It is not a "
                    "Vulkan-capable build.",
                )

            return (
                True,
                "Vulkan backend requested explicitly.",
            )

        except (
            OSError,
            subprocess.SubprocessError,
        ) as exc:
            return (
                False,
                f"Unable to verify Vulkan backend support: {exc}",
            )

    def vulkan_device_ids(self):
        return VulkanGPUManager.parse_ids(
            self.vulkan_devices_var.get().strip()
        )

    def vulkan_environment(self, visible_ids=None):
        env = os.environ.copy()

        if (
            self.backend_var.get()
            .strip()
            .lower()
            == "vulkan"
        ):
            if visible_ids is None:
                normalized = self.vulkan_device_ids()
            else:
                normalized = tuple(visible_ids)

            if normalized:
                env["GGML_VK_VISIBLE_DEVICES"] = ",".join(
                    str(item)
                    for item in normalized
                )
            else:
                env.pop(
                    "GGML_VK_VISIBLE_DEVICES",
                    None,
                )
        else:
            env.pop(
                "GGML_VK_VISIBLE_DEVICES",
                None,
            )

        return env
