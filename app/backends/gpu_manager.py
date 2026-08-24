import os
import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class VulkanGPU:
    index: int
    description: str
    memory_total: int
    memory_free: int

    @property
    def total_gib(self):
        return (
            self.memory_total / (1024 ** 3)
            if self.memory_total
            else 0.0
        )

    @property
    def free_gib(self):
        return (
            self.memory_free / (1024 ** 3)
            if self.memory_free
            else 0.0
        )


class VulkanGPUManager:
    """
    Vulkan discovery and safe workload planning for HandyAidGUI.

    The current transcribe-cli API does not expose tensor/model splitting.
    A multi-GPU worker therefore loads one full model copy per process and
    explicitly pins that process to one physical Vulkan adapter.

    Physical Vulkan indices and transcribe-cli --device indices are NOT the
    same namespace when GGML_VK_VISIBLE_DEVICES is used. The caller should
    expose one physical GPU per worker and use --device 0 in that worker.
    """

    MEMORY_HEADER_RE = re.compile(
        r"memory:\s*"
        r"(?P<total>[0-9]+(?:\.[0-9]+)?)\s*GiB\s*total,\s*"
        r"(?P<free>[0-9]+(?:\.[0-9]+)?)\s*GiB\s*free",
        re.IGNORECASE,
    )
    KIND_RE = re.compile(
        r"kind=(?P<kind>[A-Za-z0-9_+-]+)\s+"
        r"type=(?P<type>[A-Za-z0-9_+-]+)",
        re.IGNORECASE,
    )

    @staticmethod
    def parse_ids(value: str) -> tuple[int, ...]:
        text = (value or "").strip()
        if not text:
            return ()

        if not re.fullmatch(
            r"\d+(?:\s*,\s*\d+)*",
            text,
        ):
            raise ValueError(
                "Vulkan device IDs must look like 0, 1, or 0,1."
            )

        return tuple(
            dict.fromkeys(
                int(part.strip())
                for part in text.split(",")
            )
        )

    @classmethod
    def parse_list_devices(cls, text: str) -> list[VulkanGPU]:
        """
        Parse transcribe-cli --list-devices output.

        The parser deliberately accepts only Vulkan GPU/IGPU entries, never
        the CPU entry that the CLI also reports.
        """
        lines = text.splitlines()
        devices: list[VulkanGPU] = []

        current_index = None
        current_description = None
        current_kind = None
        current_type = None
        current_total = 0
        current_free = 0

        def flush():
            nonlocal current_index
            nonlocal current_description
            nonlocal current_kind
            nonlocal current_type
            nonlocal current_total
            nonlocal current_free

            if (
                current_index is not None
                and current_description
                and current_kind
                and current_kind.lower() == "vulkan"
                and current_type
                and current_type.lower()
                in {"gpu", "igpu"}
            ):
                devices.append(
                    VulkanGPU(
                        index=current_index,
                        description=current_description,
                        memory_total=current_total,
                        memory_free=current_free,
                    )
                )

            current_index = None
            current_description = None
            current_kind = None
            current_type = None
            current_total = 0
            current_free = 0

        for raw in lines:
            line = raw.rstrip()

            header = re.match(
                r"^\s*\[(\d+)\]\s*(.+?)\s*$",
                line,
            )
            if header:
                flush()
                current_index = int(header.group(1))
                current_description = header.group(2).strip()
                continue

            kind_match = cls.KIND_RE.search(line)
            if kind_match:
                current_kind = kind_match.group("kind")
                current_type = kind_match.group("type")
                continue

            memory_match = cls.MEMORY_HEADER_RE.search(line)
            if memory_match:
                current_total = int(
                    float(memory_match.group("total"))
                    * (1024 ** 3)
                )
                current_free = int(
                    float(memory_match.group("free"))
                    * (1024 ** 3)
                )

        flush()
        return devices

    @staticmethod
    def _query_env(
        base_env: dict[str, str],
        visible_device: int,
    ) -> dict[str, str]:
        env = dict(base_env)
        env["GGML_VK_VISIBLE_DEVICES"] = str(
            visible_device
        )
        return env

    @classmethod
    def probe_all(cls, binary: str) -> list[VulkanGPU]:
        """
        Enumerate every Vulkan adapter the backend can see, with no
        GGML_VK_VISIBLE_DEVICES restriction applied.

        Used to sanity-check a saved/selected device-ID list against what
        physically exists before committing to multi-GPU balanced mode.
        Some Vulkan ICDs silently ignore an out-of-range
        GGML_VK_VISIBLE_DEVICES value instead of failing, which would
        otherwise let two different "selected" physical IDs both resolve
        to the same real (often UMA/iGPU) adapter.
        """
        flags = (
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        )

        result = subprocess.run(
            [binary, "--list-devices"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            creationflags=flags,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Could not enumerate Vulkan devices.\n\n"
                f"{(result.stdout or '')[-2500:]}"
            )

        return cls.parse_list_devices(result.stdout or "")

    @classmethod
    def probe(
        cls,
        binary: str,
        physical_ids: tuple[int, ...],
    ) -> list[VulkanGPU]:
        """
        Probe each selected physical adapter independently.

        Each probe exposes exactly one Vulkan physical device, so the first
        Vulkan device in transcribe-cli's process-local registry is guaranteed
        to correspond to the physical adapter being probed.
        """
        if not physical_ids:
            return []

        # Sanity-check against an unrestricted enumeration first. Without
        # this, a stale/corrupted multi-device selection on a single-adapter
        # (often UMA iGPU) machine would silently resolve every requested ID
        # to the same physical adapter, and the caller would go on to spawn
        # two full model workers competing for one adapter's shared memory.
        total = cls.probe_all(binary)
        real_indices = {gpu.index for gpu in total}
        requested = set(physical_ids)

        if not requested.issubset(real_indices):
            available = (
                ", ".join(
                    f"{gpu.index} = {gpu.description}" for gpu in total
                )
                or "none"
            )
            raise RuntimeError(
                f"{len(total)} Vulkan adapter(s) actually present "
                f"({available}), but device ID(s) "
                f"{sorted(requested - real_indices)} were requested and "
                "do not correspond to a real adapter."
            )

        found: list[VulkanGPU] = []
        base_env = os.environ.copy()
        flags = (
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        )

        for physical_id in physical_ids:
            env = cls._query_env(
                base_env,
                physical_id,
            )

            result = subprocess.run(
                [binary, "--list-devices"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                creationflags=flags,
                env=env,
            )

            text = result.stdout or ""
            devices = cls.parse_list_devices(text)

            if result.returncode != 0:
                raise RuntimeError(
                    f"Could not query Vulkan GPU "
                    f"{physical_id}.\n\n{text[-2500:]}"
                )

            if not devices:
                raise RuntimeError(
                    f"Vulkan GPU {physical_id} was exposed, "
                    "but transcribe-cli reported no Vulkan GPU."
                    f"\n\n{text[-2500:]}"
                )

            # Because only one Vulkan adapter is visible, the first Vulkan
            # entry identifies the physical adapter we exposed.
            gpu = devices[0]

            found.append(
                VulkanGPU(
                    index=physical_id,
                    description=gpu.description,
                    memory_total=gpu.memory_total,
                    memory_free=gpu.memory_free,
                )
            )

        return found

    @staticmethod
    def choose_workers(
        gpus: list[VulkanGPU],
        model_size_bytes: int,
        safety_ratio: float = 0.75,
        fixed_reserve_bytes: int = 512 * 1024 * 1024,
    ) -> list[VulkanGPU]:
        """
        Select GPUs that conservatively appear able to hold one full model.

        The check intentionally uses free VRAM, not total VRAM. A GPU with
        insufficient free memory is excluded rather than allowed to OOM.
        """
        if not gpus or model_size_bytes <= 0:
            return []

        safety_ratio = max(
            0.50,
            min(0.90, safety_ratio),
        )

        required = int(
            model_size_bytes / safety_ratio
        ) + fixed_reserve_bytes

        return [
            gpu
            for gpu in gpus
            if gpu.memory_free > 0
            and gpu.memory_free >= required
        ]

    @staticmethod
    def weighted_chunks(
        duration_s: float,
        gpus: list[VulkanGPU],
    ):
        """
        Weight audio duration by available VRAM.

        Equal-memory GPUs receive equal time. A GPU with twice the available
        VRAM receives approximately twice the audio duration.

        This is intentionally capacity-based. It is not a benchmark-derived
        performance model.
        """
        if duration_s <= 0 or not gpus:
            return []

        weights = [
            max(1.0, gpu.memory_free)
            for gpu in gpus
        ]
        total_weight = sum(weights)

        result = []
        start = 0.0

        for index, (gpu, weight) in enumerate(
            zip(gpus, weights)
        ):
            if index == len(gpus) - 1:
                end = duration_s
            else:
                end = (
                    start
                    + duration_s
                    * (weight / total_weight)
                )

            result.append(
                (
                    gpu,
                    start,
                    max(
                        0.01,
                        end - start,
                    ),
                )
            )
            start = end

        return result

    @staticmethod
    def format_plan(
        gpus: list[VulkanGPU],
        model_size_bytes: int,
    ) -> str:
        model_gib = (
            model_size_bytes / (1024 ** 3)
        )

        lines = [
            f"GGUF file size: {model_gib:.2f} GiB"
        ]

        for gpu in gpus:
            lines.append(
                f"GPU {gpu.index}: "
                f"{gpu.description} | "
                f"{gpu.free_gib:.2f} GiB free / "
                f"{gpu.total_gib:.2f} GiB total"
            )

        return "\n".join(lines)
