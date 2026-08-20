import os
from pathlib import Path

from app.models.gguf import read_gguf_metadata

class ModelCompatibilityMixin:
    def inspect_model_compatibility(self, model_path):
        path = Path(model_path)
        info = {
            "path": str(path),
            "architecture": None,
            "name": path.name,
            "source": None,
            "compatible": None,
            "reason": "",
            "candidates": [],
        }
        wanted = {
            "general.architecture",
            "general.name",
            "general.source.url",
            "general.source.huggingface.repository",
        }
        try:
            meta = self._read_gguf_metadata(path, wanted)
            info["architecture"] = meta.get("general.architecture")
            info["name"] = meta.get("general.name") or path.name
            info["source"] = meta.get("general.source.url") or meta.get("general.source.huggingface.repository")
        except (OSError, ValueError) as exc:
            info["reason"] = f"Could not inspect GGUF metadata: {exc}"
            return info

        name = path.name.lower()
        source = (info["source"] or "").lower()
        recommended = (
            "moss-transcribe-diarize-q4_k_m",
            "moss-transcribe-diarize-q5_k_m",
            "moss-transcribe-diarize-q6_k",
            "moss-transcribe-diarize-q8_0",
            "moss-transcribe-diarize-f16",
            "moss-transcribe-diarize-bf16",
        )
        is_crispasr = (
            "crispasr" in source
            or "moss-transcribe-diarize-0.9b-q4_k" in name
        )
        is_handy_named = any(x in name for x in recommended)
        # Handy's published MOSS GGUFs identify the family in filenames, while
        # the GGUF general.architecture field may be the generic "moss" value.
        # Do not require the backend-specific internal label here; the previous
        # checker incorrectly rejected valid Handy models reporting "moss".
        is_moss_architecture = info["architecture"] in {"moss", "moss_transcribe_diarize"}

        if is_crispasr:
            info["compatible"] = False
            info["reason"] = (
                "This is the 0.9B q4_k MOSS conversion distributed for CrispASR. "
                "It is not the MOSS GGUF family documented for the Handy transcribe.cpp executable."
            )
        elif is_handy_named and is_moss_architecture:
            info["compatible"] = True
            info["reason"] = "GGUF filename and architecture match the Handy transcribe.cpp MOSS Transcribe-Diarize model family."
        elif is_moss_architecture:
            info["compatible"] = None
            info["reason"] = "GGUF architecture is MOSS-compatible; backend validation will confirm the exact conversion."
        elif info["architecture"]:
            info["compatible"] = False
            info["reason"] = f"GGUF architecture '{info['architecture']}' is not moss_transcribe_diarize."
        else:
            info["reason"] = "No GGUF architecture metadata was found. Backend validation is required."

        try:
            for candidate in sorted(path.parent.glob("*.gguf")):
                c = candidate.name.lower()
                if any(x in c for x in recommended):
                    info["candidates"].append(str(candidate))
        except OSError:
            pass
        return info

    def update_model_status(self):
        if not hasattr(self, "model_status_label"):
            return
        model = self.model_path_var.get().strip()
        if not model or not Path(model).is_file():
            self.model_status_label.config(text="Model compatibility: not checked", foreground=self.MUTED)
            self.model_info = {}
            return
        info = self.inspect_model_compatibility(model)
        self.model_info = info
        self.compatible_model_candidates = info.get("candidates", [])
        if info["compatible"] is False:
            self.model_status_label.config(text="Model compatibility: INCOMPATIBLE — see Diagnose model", foreground=self.DANGER)
        elif info["compatible"] is True:
            self.model_status_label.config(text="Model compatibility: compatible MOSS GGUF", foreground=self.SUCCESS)
        else:
            self.model_status_label.config(text="Model compatibility: backend validation required", foreground=self.WARNING)

    def diagnose_model(self):
        model = self.model_path_var.get().strip()
        if not model or not Path(model).is_file():
            messagebox.showerror("Model diagnosis", "Select a valid GGUF model first.")
            return
        info = self.inspect_model_compatibility(model)
        self.model_info = info
        self.compatible_model_candidates = info.get("candidates", [])
        self.update_model_status()
        lines = [
            f"File: {info['name']}",
            f"Architecture: {info.get('architecture') or 'unknown'}",
            f"Source: {info.get('source') or 'unknown'}",
            "",
            info.get("reason", "No assessment available."),
        ]
        if info.get("candidates"):
            lines.extend(["", "Handy/transcribe.cpp-style models found in the same folder:"])
            lines.extend(f"• {item}" for item in info["candidates"][:8])
        messagebox.showinfo("MOSS model diagnosis", "\n".join(lines))

    def detect_default_model(self):
        candidates = []
        search_dirs = [Path.cwd(), Path(__file__).resolve().parent]
        seen = set()
        for directory in search_dirs:
            try:
                directory = directory.resolve()
            except OSError:
                continue
            if directory in seen or not directory.exists():
                continue
            seen.add(directory)
            try:
                for path in sorted(directory.iterdir()):
                    lower = path.name.lower()
                    if path.is_file() and (lower.endswith(".gguf") or (lower.endswith(".bin") and "moss" in lower)):
                        candidates.append(path)
            except OSError:
                continue
        if candidates and not self.model_path_var.get().strip():
            self.model_path_var.set(str(candidates[0].resolve()))
        if candidates:
            self.status_var.set(f"Model found: {candidates[0].name}")
        elif not self.model_path_var.get().strip():
            self.status_var.set("No MOSS model detected")
        self.refresh_summaries()
        self.update_capabilities()
        self.update_backend_ui()
        self.update_model_status()
