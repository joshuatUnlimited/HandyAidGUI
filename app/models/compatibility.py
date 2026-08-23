import os
from pathlib import Path
from tkinter import messagebox

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
            # This is the only real gate: the file itself isn't a readable
            # GGUF. Which architecture it declares is not this GUI's call --
            # transcribe-cli is the actual authority on what it can run.
            info["compatible"] = False
            info["reason"] = f"Could not read this file as a GGUF model: {exc}"
            return info

        info["compatible"] = True

        name = path.name.lower()
        source = (info["source"] or "").lower()
        moss_markers = (
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
        is_moss_named = any(x in name for x in moss_markers)
        # Handy's published MOSS GGUFs identify the family in filenames, while
        # the GGUF general.architecture field may be the generic "moss" value.
        is_moss_architecture = info["architecture"] in {"moss", "moss_transcribe_diarize"}

        if is_crispasr:
            info["reason"] = (
                "This looks like the 0.9B q4_k MOSS conversion distributed for CrispASR, "
                "not the MOSS GGUF family this GUI was originally documented against. "
                "It may still load fine -- if transcribe-cli rejects it, the error will "
                "show in the Process Log."
            )
        elif is_moss_named and is_moss_architecture:
            info["reason"] = "Filename and architecture match the documented MOSS Transcribe-Diarize family."
        elif info["architecture"]:
            info["reason"] = (
                f"Architecture: '{info['architecture']}'. Not the MOSS family this GUI "
                "was originally built around, but any GGUF transcribe-cli's --help "
                "advertises support for should work here."
            )
        else:
            info["reason"] = "No architecture metadata found in the GGUF. transcribe-cli will validate it at run time."

        try:
            for candidate in sorted(path.parent.glob("*.gguf")):
                c = candidate.name.lower()
                if any(x in c for x in moss_markers):
                    info["candidates"].append(str(candidate))
        except OSError:
            pass
        return info

    def update_model_status(self):
        if not hasattr(self, "model_status_label"):
            return
        model = self.model_path_var.get().strip()
        if not model or not Path(model).is_file():
            self.model_status_label.config(text="Model check: not checked", foreground=self.MUTED)
            self.model_info = {}
            return
        info = self.inspect_model_compatibility(model)
        self.model_info = info
        self.compatible_model_candidates = info.get("candidates", [])
        if info["compatible"] is False:
            self.model_status_label.config(text="Model check: UNREADABLE — see Diagnose model", foreground=self.DANGER)
        else:
            arch = info.get("architecture") or "unknown architecture"
            self.model_status_label.config(text=f"Model check: valid GGUF ({arch})", foreground=self.SUCCESS)

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
            lines.extend(["", "MOSS-family GGUF(s) also found in the same folder:"])
            lines.extend(f"• {item}" for item in info["candidates"][:8])
        messagebox.showinfo("Model diagnosis", "\n".join(lines))

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
                    if path.is_file() and path.name.lower().endswith(".gguf"):
                        candidates.append(path)
            except OSError:
                continue
        if candidates and not self.model_path_var.get().strip():
            self.model_path_var.set(str(candidates[0].resolve()))
        if candidates:
            self.status_var.set(f"Model found: {candidates[0].name}")
        elif not self.model_path_var.get().strip():
            self.status_var.set("No GGUF model detected")
        self.refresh_summaries()
        self.update_capabilities()
        self.update_backend_ui()
        self.update_model_status()
