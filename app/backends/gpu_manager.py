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

    @classmethod
    def find_likely_shared_memory_duplicates(
        cls,
        gpus: list[VulkanGPU],
        tolerance: float = 0.02,
    ) -> list[VulkanGPU]:
        """
        Flag (do NOT remove) GPUs whose description and memory_total
        both match another selected GPU closely enough that they might
        be one physical shared-memory device registered twice (e.g.
        duplicate Vulkan ICD registration on an iGPU) rather than two
        independent memory pools.

        This is deliberately informational only. An earlier version of
        this method excluded the duplicate from scheduling, but the
        only signals --list-devices actually provides here (description
        text, memory_total) can't reliably tell that case apart from two
        genuinely separate, IDENTICAL dedicated GPUs — a matched
        multi-GPU pair (e.g. two RTX 4090s bought specifically for this
        kind of workload) reports the same description and the same
        memory_total too. Auto-excluding on this signal would silently
        break real parallelism for exactly the hardware this feature is
        for. So this only warns; it never changes what choose_workers
        returns. If your transcribe-cli build's --list-devices output
        has an actual shared-memory/uma marker, send the exact line and
        this can be tightened to use that instead of guessing.
        """
        groups: dict[int, list[VulkanGPU]] = {}
        for gpu in gpus:
            bucket = None
            for key, members in groups.items():
                ref = members[0]
                if (
                    ref.description == gpu.description
                    and ref.memory_total
                    and abs(gpu.memory_total - ref.memory_total) <= ref.memory_total * tolerance
                ):
                    bucket = key
                    break
            if bucket is None:
                bucket = gpu.index
                groups[bucket] = []
            groups[bucket].append(gpu)

        flagged: list[VulkanGPU] = []
        for members in groups.values():
            if len(members) > 1:
                flagged.extend(sorted(members, key=lambda g: g.index)[1:])

        flagged.sort(key=lambda g: g.index)
        return flagged

    @staticmethod
    def weighted_chunks(
        duration_s: float,
        gpus: list[VulkanGPU],
    ):
        """
        Legacy API, kept for backward compatibility with any other
        caller (e.g. a GPU control panel preview) that may already
        depend on this exact signature/behavior. AI SBU itself now
        uses weighted_boundaries + sequential_windows instead, which
        additionally bounds each transcribe-cli call's audio length —
        this method does NOT provide that bound, so don't route new
        transcription work through it.

        Weight audio duration by available VRAM. Equal-memory GPUs
        receive equal time; a GPU with twice the available VRAM
        receives approximately twice the audio duration.
        """
        if duration_s <= 0 or not gpus:
            return []

        weights = [max(1.0, gpu.memory_free) for gpu in gpus]
        total_weight = sum(weights)

        result = []
        start = 0.0
        for index, (gpu, weight) in enumerate(zip(gpus, weights)):
            if index == len(gpus) - 1:
                end = duration_s
            else:
                end = start + duration_s * (weight / total_weight)
            result.append((gpu, start, max(0.01, end - start)))
            start = end
        return result

    @staticmethod
    def weighted_boundaries(duration_s: float, gpus: list[VulkanGPU]):
        """
        Nominal (non-overlapping) time boundaries per GPU, weighted by
        free memory. This only decides which GPU EXECUTES which small
        window (see sequential_windows) — it is no longer responsible
        for bounding how much audio any single transcribe-cli call
        receives, which is the actual crash-safety property.
        """
        if duration_s <= 0 or not gpus:
            return []

        weights = [max(1.0, gpu.memory_free) for gpu in gpus]
        total = sum(weights)

        bounds = [0.0]
        running = 0.0
        for w in weights[:-1]:
            running += duration_s * (w / total)
            bounds.append(running)
        bounds.append(duration_s)

        return [(gpu, bounds[i], bounds[i + 1]) for i, gpu in enumerate(gpus)]

    @staticmethod
    def sequential_windows(
        start: float,
        end: float,
        window_s: float = 25.0,
        overlap_s: float = 4.0,
    ):
        """
        Split [start, end] into small overlapping windows, independent
        of GPU count. This bounds how much audio any single
        transcribe-cli call ever receives — the actual fix for
        "failed to allocate ... buffer of size ~19GB", since that
        allocation scales with input length, not model size or GPU
        count. A single-GPU (or CPU) machine needs this exactly as
        much as a multi-GPU one does.

        Each window is returned as
        (window_start, window_end, nominal_start, nominal_end) —
        window_start/end include the overlap padding actually sent to
        the model for context; nominal_start/end is the boundary this
        window OWNS in the stitched output (used by ai_sbu_core to
        avoid duplicate/dropped text at cut points).
        """
        if window_s <= overlap_s:
            raise ValueError("window_s must be greater than overlap_s")
        if end <= start:
            return []

        windows = []
        nominal_start = start
        step = window_s - overlap_s

        while nominal_start < end:
            nominal_end = min(nominal_start + step, end)
            win_start = (
                max(start, nominal_start - overlap_s)
                if nominal_start > start
                else nominal_start
            )
            win_end = (
                min(end, nominal_end + overlap_s)
                if nominal_end < end
                else nominal_end
            )
            windows.append((win_start, win_end, nominal_start, nominal_end))
            if nominal_end >= end:
                break
            nominal_start = nominal_end

        return windows

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
