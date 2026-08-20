import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VulkanGPU:
    index: int
    description: str
    memory_total: int
    memory_free: int

    @property
    def total_gib(self):
        return self.memory_total / (1024 ** 3) if self.memory_total else 0.0

    @property
    def free_gib(self):
        return self.memory_free / (1024 ** 3) if self.memory_free else 0.0


class VulkanGPUManager:
    """
    GPU discovery and safe workload balancing for HandyAidGUI.

    The current transcribe-cli API selects one compute device per process.
    It does not expose a model tensor-split control. Therefore this manager
    deliberately uses one process per selected Vulkan physical GPU rather than
    inventing unsupported --tensor-split/--split-mode arguments.

    Each worker loads the model on exactly one Vulkan GPU. This is useful when
    a workload can fit independently on each adapter and the goal is balanced
    throughput. It is NOT a way to make a model larger than the smallest GPU.
    """

    DEVICE_HEADER_RE = re.compile(
        r"^\s*\[(?P<index>\d+)\]\s*(?P<description>.+?)\s*$",
        re.MULTILINE,
    )
    MEMORY_RE = re.compile(
        r"memory:\s*"
        r"(?P<total>[0-9]+(?:\.[0-9]+)?)\s*GiB\s*total,\s*"
        r"(?P<free>[0-9]+(?:\.[0-9]+)?)\s*GiB\s*free",
        re.IGNORECASE,
    )
    KIND_RE = re.compile(
        r"kind=(?P<kind>[A-Za-z0-9_+-]+)\s+type=(?P<type>[A-Za-z0-9_+-]+)",
        re.IGNORECASE,
    )

    @staticmethod
    def parse_ids(value: str) -> tuple[int, ...]:
        text = (value or "").strip()
        if not text:
            return ()
        if not re.fullmatch(r"\d+(?:\s*,\s*\d+)*", text):
            raise ValueError("Vulkan device IDs must look like 0, 1, or 0,1.")
        return tuple(dict.fromkeys(int(part.strip()) for part in text.split(",")))

    @staticmethod
    def parse_list_devices(text: str) -> list[VulkanGPU]:
        lines = text.splitlines()
        devices: list[VulkanGPU] = []
        current_index = None
        current_description = None
        current_kind = None
        current_type = None
        current_total = 0
        current_free = 0

        def flush():
            nonlocal current_index, current_description, current_kind
            nonlocal current_type, current_total, current_free
            if (
                current_index is not None
                and current_description
                and current_kind
                and current_kind.lower() == "vulkan"
                and current_type
                and current_type.lower() in {"gpu", "igpu"}
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
            header = re.match(r"^\s*\[(\d+)\]\s*(.+?)\s*$", line)
            if header:
                flush()
                current_index = int(header.group(1))
                current_description = header.group(2).strip()
                continue

            kind_match = VulkanGPUManager.KIND_RE.search(line)
            if kind_match:
                current_kind = kind_match.group("kind")
                current_type = kind_match.group("type")
                continue

            memory_match = VulkanGPUManager.MEMORY_RE.search(line)
            if memory_match:
                current_total = int(float(memory_match.group("total")) * (1024 ** 3))
                current_free = int(float(memory_match.group("free")) * (1024 ** 3))

        flush()
        return devices

    @staticmethod
    def _query_env(base_env: dict[str, str], visible_device: int | None) -> dict[str, str]:
        env = dict(base_env)
        if visible_device is None:
            env.pop("GGML_VK_VISIBLE_DEVICES", None)
        else:
            env["GGML_VK_VISIBLE_DEVICES"] = str(visible_device)
        return env

    @classmethod
    def probe(cls, binary: str, physical_ids: tuple[int, ...]) -> list[VulkanGPU]:
        """
        Probe each selected physical Vulkan adapter independently.

        Running --list-devices once per physical ID is intentional: the
        transcribe runtime has a process-local compute-device registry, and its
        --device index is not the same thing as the Vulkan physical index.
        Restricting GGML_VK_VISIBLE_DEVICES to one adapter lets us identify the
        adapter unambiguously without relying on registry ordering.
        """
        if not physical_ids:
            return []

        found: list[VulkanGPU] = []
        base_env = os.environ.copy()
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        for physical_id in physical_ids:
            env = cls._query_env(base_env, physical_id)
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
            if result.returncode != 0 or not devices:
                raise RuntimeError(
                    f"Could not query Vulkan GPU {physical_id}.\n\n"
                    f"{text[-2500:]}"
                )

            gpu = next(
                (item for item in devices if item.memory_total or item.memory_free),
                devices[0],
            )
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
        Return GPUs that conservatively appear able to hold a full copy of the
        model.

        A worker does not share model weights with another worker, so every
        worker must individually fit the model. Unknown memory is treated as
        unsafe for automatic multi-GPU scheduling.
        """
        if not gpus:
            return []

        required = int(model_size_bytes / max(0.01, safety_ratio)) + fixed_reserve_bytes
        eligible = [
            gpu
            for gpu in gpus
            if gpu.memory_free > 0 and gpu.memory_free >= required
        ]
        return eligible

    @staticmethod
    def weighted_chunks(duration_s: float, gpus: list[VulkanGPU]):
        """
        Split audio by currently available VRAM.

        Equal-capacity GPUs receive equal-duration chunks. A GPU with twice the
        free VRAM receives roughly twice the audio. This is a capacity-oriented
        distribution, not a claim that the adapters have equal throughput.
        """
        if duration_s <= 0 or not gpus:
            return []

        weights = [max(1.0, gpu.memory_free) for gpu in gpus]
        total = sum(weights)
        result = []
        start = 0.0

        for index, (gpu, weight) in enumerate(zip(gpus, weights)):
            if index == len(gpus) - 1:
                end = duration_s
            else:
                end = start + duration_s * (weight / total)
            result.append((gpu, start, max(0.01, end - start)))
            start = end

        return result

    @staticmethod
    def format_plan(gpus: list[VulkanGPU], model_size_bytes: int) -> str:
        model_gib = model_size_bytes / (1024 ** 3)
        lines = [f"Model file size: {model_gib:.2f} GiB"]
        for gpu in gpus:
            lines.append(
                f"GPU {gpu.index}: {gpu.description} | "
                f"{gpu.free_gib:.2f} GiB free / {gpu.total_gib:.2f} GiB total"
            )
        return "\n".join(lines)
