"""
app/gui/gpu_panel.py

GPU control panel for HandyAidGUI.

Lets the user see every Vulkan-capable GPU on the system, enable/disable
individual GPUs, and control how a transcription job's workload is split
across them -- without touching the command line or hand-editing
GGML_VK_VISIBLE_DEVICES.

Designed as a mixin (`GPUControlPanelMixin`) so it composes into the main
app window class without introducing tight coupling to it. The host class
is expected to provide, before calling `build_gpu_tab()`:

    self.gpu_manager        Object with:
                               .enumerate_all(binary) -> list[GPUInfo]
                               .probe(binary, index) -> GPUInfo   (optional;
                                   enables cheap periodic VRAM refresh
                                   without a full re-enumeration)
    self.transcribe_binary  Path to the transcribe-cli binary (str).
    self.vulkan_devices_var A tk.StringVar already read by
                             command_builder.py / vulkan.py. This mixin
                             only ever *writes* to it -- nothing
                             downstream needs to change.
    self.settings            Optional. Object with .get(key, default) /
                             .set(key, value). Persistence degrades
                             gracefully to "nothing persists" if absent.
    self.get_model_size_bytes()
                             Optional callable returning the currently
                             selected model's size in bytes. Without it,
                             eligibility checks fall back to "does this
                             GPU have any free VRAM at all" instead of a
                             real fits/doesn't-fit judgement.

Run this file directly (`python gpu_panel.py`) for a self-contained demo
window with fake GPUs -- see the bottom of the file.

--------------------------------------------------------------------------
Design notes / lessons carried over from an earlier debugging pass on this
same feature -- kept here so the next person editing this file doesn't
reintroduce the same three bugs:

1. Populate every per-GPU data structure (`_gpu_infos`, `_gpu_states`)
   COMPLETELY before any widget that reads them is built. A card that
   reads `self._gpu_states[index]` while that dict is only half-populated
   is a KeyError waiting to happen the first time scan results arrive in
   a different order than the UI was built in.

2. Never compare a manual weight (0-100, "% of duration") against a raw
   free-VRAM byte count. They're different units. Free-VRAM bytes are
   only ever used as *relative* weights among Auto-mode GPUs, after
   manual shares have already been taken off the top in percentage
   space. See `compute_weighted_chunks()`.

3. The safety-ratio slider is easy to label backwards. The formula is:

       required_vram_bytes = model_size_bytes / safety_ratio

   so a LOWER ratio means a HIGHER required_vram_bytes threshold, which
   means MORE headroom is demanded and the check is MORE conservative.
   A ratio approaching 1.0 means "the model just barely needs to fit,
   no headroom" -- i.e. permissive. The UI labels and the live help text
   in `_update_safety_help_text()` both spell this out explicitly so it
   can't quietly drift out of sync with the math again.

This module also intentionally duplicates the weighting math instead of
importing it from app/backends/gpu_manager.py, purely so it can run
standalone (see the demo at the bottom) without the rest of the app's
import graph. In the real app, wire `compute_weighted_chunks()` to be the
same function `gpu_manager.weighted_chunks()` calls (or have one import
the other) -- do not maintain two independent copies of this math. That
divergence is exactly how the manual/auto weight-scale bug happened last
time.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MB = 1024 ** 2
GB = 1024 ** 3

DEFAULT_SAFETY_RATIO = 0.85       # required = model_size / safety_ratio
DEFAULT_RESERVE_MB = 512          # fixed VRAM held back per GPU, beyond the ratio
# Must match the clamp in app/backends/gpu_manager.py's choose_workers().
# Keeping these in sync is exactly what the module docstring warns about;
# until the two implementations are consolidated, treat this pair of
# constants as the single source of truth for the clamp bounds.
SAFETY_RATIO_MIN = 0.50
SAFETY_RATIO_MAX = 0.90

SCAN_POLL_INTERVAL_MS = 150       # how often we check the rescan queue
LIVE_REFRESH_INTERVAL_MS = 5000   # how often we cheaply re-probe VRAM

COLOR_FIT = "#1b8a3c"
COLOR_INSUFFICIENT = "#b3261e"
COLOR_DISABLED = "#8a8a8a"
COLOR_UNKNOWN = "#a67c00"
COLOR_BAR_BG = "#3a3a3a"
COLOR_BAR_USED = "#5b7fff"
COLOR_BAR_FREE = "#2fbf71"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Hardware facts for one physical Vulkan device, as reported by
    gpu_manager.enumerate_all() / .probe(). Refreshed wholesale on
    rescan, refreshed partially (vram_free only) on the periodic live
    refresh.
    """
    index: int
    name: str
    vram_total: int  # bytes
    vram_free: int   # bytes


@dataclass
class GPUState:
    """Per-GPU UI/user state. Deliberately kept separate from GPUInfo so
    a rescan or a live VRAM refresh can update hardware facts without
    clobbering what the user has chosen.
    """
    enabled: bool = True
    mode: str = "auto"                 # "auto" | "manual"
    manual_weight_pct: float = 0.0     # 0-100, meaningful only in manual mode
    eligible: bool = True
    computed_weight_pct: float = 0.0   # last-solved effective share (for display)


# ---------------------------------------------------------------------------
# Pure weighting logic -- no Tk, no I/O, fully unit-testable in isolation
# ---------------------------------------------------------------------------

def compute_weighted_chunks(
    infos: dict[int, GPUInfo],
    states: dict[int, GPUState],
    safety_ratio: float,
    reserve_bytes: int,
    model_size_bytes: Optional[int],
) -> dict[int, dict]:
    """Given hardware facts and user choices, decide which GPUs are
    eligible and what % share of the workload each one gets.

    Returns ``{index: {"eligible": bool, "weight_pct": float}}`` for
    every GPU in `infos`, including disabled/ineligible ones
    (weight_pct is always 0.0 for those).

    Rules:
      - Disabled or ineligible GPUs always get 0%.
      - Manual-mode GPUs get their user-set % share, taken off the top.
      - If manual shares alone exceed 100%, they're renormalized
        proportionally down to sum to 100%, and every Auto GPU gets 0%
        (there's nothing left for them).
      - Otherwise, the remainder (100% - sum of manual shares) is split
        across enabled+eligible Auto-mode GPUs, weighted by each one's
        free VRAM after the reserve is subtracted.
      - If there are no Auto GPUs to absorb the remainder, manual shares
        are scaled *up* to fill 100% instead of leaving work unassigned.
      - If the Auto GPUs' combined free VRAM is 0 (all pinned to the
        reserve floor), the remainder is split evenly among them rather
        than dividing by zero.
    """
    result = {index: {"eligible": False, "weight_pct": 0.0} for index in infos}

    required_bytes: Optional[float] = None
    if model_size_bytes:
        # Clamped to the same [MIN, MAX] range gpu_manager.choose_workers()
        # enforces server-side -- a ratio outside this range would be
        # silently overridden there, so eligibility computed here must
        # agree with what actually runs.
        safe_ratio = max(SAFETY_RATIO_MIN, min(SAFETY_RATIO_MAX, safety_ratio))
        required_bytes = model_size_bytes / safe_ratio

    usable: dict[int, int] = {}  # index -> free bytes after reserve, enabled GPUs only
    for index, info in infos.items():
        state = states[index]
        free_after_reserve = max(info.vram_free - reserve_bytes, 0)

        if required_bytes is None:
            # No model loaded yet -- can't judge "does it fit", so don't
            # block on that basis. Just require some free headroom.
            eligible = free_after_reserve > 0
        else:
            eligible = free_after_reserve >= required_bytes

        result[index]["eligible"] = eligible

        if state.enabled and eligible:
            usable[index] = free_after_reserve

    manual_indices = [i for i in usable if states[i].mode == "manual"]
    auto_indices = [i for i in usable if states[i].mode != "manual"]
    manual_total = sum(states[i].manual_weight_pct for i in manual_indices)

    if manual_total > 100.0 + 1e-9:
        scale = 100.0 / manual_total if manual_total else 0.0
        for i in manual_indices:
            result[i]["weight_pct"] = states[i].manual_weight_pct * scale
        for i in auto_indices:
            result[i]["weight_pct"] = 0.0
        return result

    for i in manual_indices:
        result[i]["weight_pct"] = states[i].manual_weight_pct

    remainder = max(100.0 - manual_total, 0.0)

    if auto_indices:
        vram_total = sum(usable[i] for i in auto_indices)
        if vram_total > 0:
            for i in auto_indices:
                result[i]["weight_pct"] = remainder * (usable[i] / vram_total)
        else:
            share = remainder / len(auto_indices)
            for i in auto_indices:
                result[i]["weight_pct"] = share
    elif manual_indices and remainder > 1e-9 and manual_total > 0:
        scale = 100.0 / manual_total
        for i in manual_indices:
            result[i]["weight_pct"] = states[i].manual_weight_pct * scale

    return result


# ---------------------------------------------------------------------------
# One GPU's row -- pure presentation, owns no state of its own beyond
# what's needed to avoid re-triggering its own callbacks
# ---------------------------------------------------------------------------

class _GPUCard:
    def __init__(
        self,
        parent: tk.Widget,
        index: int,
        on_enabled_toggle: Callable[[int, bool], None],
        on_mode_change: Callable[[int, str], None],
        on_weight_change: Callable[[int, float], None],
    ):
        self.index = index
        self._on_enabled_toggle = on_enabled_toggle
        self._on_mode_change = on_mode_change
        self._on_weight_change = on_weight_change
        self._last_frac_used = 0.0

        self.frame = ttk.Frame(parent, relief="groove", borderwidth=1, padding=8)
        self.frame.columnconfigure(1, weight=1)

        self.enabled_var = tk.BooleanVar(value=True)
        self.mode_var = tk.StringVar(value="auto")
        self.weight_var = tk.DoubleVar(value=0.0)

        self.enabled_check = ttk.Checkbutton(
            self.frame,
            variable=self.enabled_var,
            command=lambda: self._on_enabled_toggle(self.index, self.enabled_var.get()),
        )
        self.enabled_check.grid(row=0, column=0, rowspan=3, sticky="n", padx=(0, 8))

        self.name_label = ttk.Label(self.frame, text="", font=("TkDefaultFont", 10, "bold"))
        self.name_label.grid(row=0, column=1, sticky="w")

        self.pill_label = tk.Label(
            self.frame, text="", padx=8, pady=1, font=("TkDefaultFont", 8, "bold"),
        )
        self.pill_label.grid(row=0, column=2, sticky="e", padx=4)

        self.vram_canvas = tk.Canvas(self.frame, height=16, highlightthickness=0, bg=COLOR_BAR_BG)
        self.vram_canvas.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(2, 6))
        self.vram_canvas.bind("<Configure>", lambda e: self._draw_bar(self._last_frac_used))

        self.vram_text = ttk.Label(self.frame, text="", foreground="#666")
        self.vram_text.grid(row=2, column=1, columnspan=2, sticky="w")

        mode_row = ttk.Frame(self.frame)
        mode_row.grid(row=3, column=1, columnspan=2, sticky="ew", pady=(6, 0))
        mode_row.columnconfigure(2, weight=1)

        ttk.Radiobutton(
            mode_row, text="Auto", value="auto", variable=self.mode_var,
            command=lambda: self._on_mode_change(self.index, self.mode_var.get()),
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode_row, text="Manual", value="manual", variable=self.mode_var,
            command=lambda: self._on_mode_change(self.index, self.mode_var.get()),
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.weight_scale = ttk.Scale(
            mode_row, from_=0, to=100, orient="horizontal", variable=self.weight_var,
            command=lambda v: self._on_weight_change(self.index, float(v)),
        )
        self.weight_scale.grid(row=0, column=2, sticky="ew", padx=8)

        self.weight_label = ttk.Label(mode_row, text="0%", width=14, anchor="e")
        self.weight_label.grid(row=0, column=3, sticky="e")

    def render(self, info: GPUInfo, state: GPUState) -> None:
        self.name_label.configure(text=f"[{info.index}] {info.name}")

        # Sync vars without re-firing their own commands.
        if self.enabled_var.get() != state.enabled:
            self.enabled_var.set(state.enabled)
        if self.mode_var.get() != state.mode:
            self.mode_var.set(state.mode)

        target_weight = state.manual_weight_pct if state.mode == "manual" else state.computed_weight_pct
        if abs(self.weight_var.get() - target_weight) > 0.01:
            self.weight_var.set(target_weight)

        self.weight_scale.state(
            ["!disabled"] if (state.enabled and state.mode == "manual") else ["disabled"]
        )

        used = max(info.vram_total - info.vram_free, 0)
        frac_used = (used / info.vram_total) if info.vram_total else 0.0
        self._draw_bar(frac_used)

        self.vram_text.configure(
            text=f"{info.vram_free / GB:.2f} GB free / {info.vram_total / GB:.2f} GB total"
        )

        if not state.enabled:
            self._set_pill("DISABLED", COLOR_DISABLED)
        elif state.eligible:
            self._set_pill("FITS MODEL", COLOR_FIT)
        else:
            self._set_pill("INSUFFICIENT VRAM", COLOR_INSUFFICIENT)

        suffix = "" if state.mode == "manual" else " (auto)"
        self.weight_label.configure(text=f"{target_weight:.0f}%{suffix}")

    def _draw_bar(self, frac_used: float) -> None:
        self._last_frac_used = frac_used
        self.vram_canvas.delete("all")
        width = max(self.vram_canvas.winfo_width(), 1)
        height = 16
        used_w = int(width * min(max(frac_used, 0.0), 1.0))
        self.vram_canvas.create_rectangle(0, 0, width, height, fill=COLOR_BAR_BG, width=0)
        self.vram_canvas.create_rectangle(0, 0, used_w, height, fill=COLOR_BAR_USED, width=0)
        self.vram_canvas.create_rectangle(used_w, 0, width, height, fill=COLOR_BAR_FREE, width=0)

    def _set_pill(self, text: str, color: str) -> None:
        self.pill_label.configure(text=text, bg=color, fg="white")


# ---------------------------------------------------------------------------
# The mixin
# ---------------------------------------------------------------------------

class GPUControlPanelMixin:
    """Compose this into the main window class. Call `build_gpu_tab(parent)`
    once, after `self.vulkan_devices_var` (and ideally `self.gpu_manager`,
    `self.transcribe_binary`, `self.settings`) already exist.
    """

    # -- lifecycle ---------------------------------------------------------

    def build_gpu_tab(self, parent: tk.Widget) -> ttk.Frame:
        # Every data structure below must exist, fully, before a single
        # widget is built -- see design note (1) at the top of the file.
        self._gpu_infos: dict[int, GPUInfo] = {}
        self._gpu_states: dict[int, GPUState] = {}
        self._gpu_cards: dict[int, _GPUCard] = {}

        self._gpu_scan_queue: "queue.Queue" = queue.Queue()
        self._gpu_scan_lock = threading.Lock()
        self._gpu_scan_in_progress = False
        self._gpu_scan_poll_job: Optional[str] = None

        self._gpu_live_queue: "queue.Queue" = queue.Queue()
        self._gpu_live_refresh_job: Optional[str] = None
        self._gpu_live_poll_job: Optional[str] = None

        self.safety_ratio_var = tk.DoubleVar(
            value=self._settings_get("gpu.safety_ratio", DEFAULT_SAFETY_RATIO)
        )
        self.vram_reserve_mb_var = tk.IntVar(
            value=self._settings_get("gpu.reserve_mb", DEFAULT_RESERVE_MB)
        )
        self._gpu_status_var = tk.StringVar(value="No scan yet.")
        self._safety_help_var = tk.StringVar()

        root = ttk.Frame(parent)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        self._gpu_tab_root = root

        self._build_global_controls(root).grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        ttk.Label(root, textvariable=self._gpu_status_var, foreground="#666").grid(
            row=1, column=0, sticky="w", padx=12
        )
        self._build_card_list(root).grid(row=2, column=0, sticky="nsew", padx=8, pady=8)

        root.bind("<Destroy>", self._on_gpu_tab_destroy, add="+")

        self._update_safety_help_text()
        self.rescan_gpus()
        self._schedule_live_refresh()

        return root

    def _on_gpu_tab_destroy(self, event: tk.Event) -> None:
        if event.widget is not self._gpu_tab_root:
            return
        for attr in ("_gpu_scan_poll_job", "_gpu_live_refresh_job", "_gpu_live_poll_job"):
            job = getattr(self, attr, None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
                setattr(self, attr, None)

    # -- settings passthrough (works with or without a real settings object)

    def _settings_get(self, key: str, default):
        settings = getattr(self, "settings", None)
        if settings is None:
            return default
        try:
            return settings.get(key, default)
        except Exception:
            return default

    def _settings_set(self, key: str, value) -> None:
        settings = getattr(self, "settings", None)
        if settings is None:
            return
        try:
            settings.set(key, value)
        except Exception:
            pass

    def _get_model_size_bytes(self) -> Optional[int]:
        getter = getattr(self, "get_model_size_bytes", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return None
        return None

    # -- global controls -----------------------------------------------

    def _build_global_controls(self, parent: tk.Widget) -> ttk.Labelframe:
        frame = ttk.Labelframe(parent, text="GPU Settings")
        frame.columnconfigure(1, weight=1)

        self._rescan_btn = ttk.Button(frame, text="Rescan GPUs", command=self.rescan_gpus)
        self._rescan_btn.grid(row=0, column=0, padx=4, pady=4, sticky="w")

        ttk.Button(frame, text="Reset to Automatic", command=self.reset_gpu_settings_to_auto).grid(
            row=0, column=1, padx=4, pady=4, sticky="w"
        )

        ttk.Label(frame, text="Safety margin:").grid(row=1, column=0, sticky="w", padx=4)
        ratio_row = ttk.Frame(frame)
        ratio_row.grid(row=1, column=1, columnspan=2, sticky="ew", padx=4)
        ratio_row.columnconfigure(1, weight=1)

        ttk.Label(ratio_row, text="Conservative").grid(row=0, column=0)
        ttk.Scale(
            ratio_row, from_=SAFETY_RATIO_MIN, to=SAFETY_RATIO_MAX, orient="horizontal",
            variable=self.safety_ratio_var, command=self._on_safety_ratio_changed,
        ).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(ratio_row, text="Permissive").grid(row=0, column=2)

        ttk.Label(frame, textvariable=self._safety_help_var, foreground="#666", wraplength=520).grid(
            row=2, column=0, columnspan=3, sticky="w", padx=4
        )

        ttk.Label(frame, text="Reserve per GPU (MB):").grid(
            row=3, column=0, sticky="w", padx=4, pady=(4, 8)
        )
        reserve_spin = ttk.Spinbox(
            frame, from_=0, to=16384, increment=128, textvariable=self.vram_reserve_mb_var,
            width=8, command=self._on_reserve_changed,
        )
        reserve_spin.grid(row=3, column=1, sticky="w", padx=4, pady=(4, 8))
        reserve_spin.bind("<Return>", lambda e: self._on_reserve_changed())
        reserve_spin.bind("<FocusOut>", lambda e: self._on_reserve_changed())

        return frame

    def _update_safety_help_text(self) -> None:
        ratio = self.safety_ratio_var.get()
        model_size = self._get_model_size_bytes()
        base = (
            f"Ratio {ratio:.2f} \u2014 lower means more headroom required (safer, "
            f"stricter), higher means tighter fit allowed (riskier, more permissive)."
        )
        if model_size:
            required_gb = (model_size / max(ratio, 0.01)) / GB
            self._safety_help_var.set(f"{base} A GPU currently needs \u2265 {required_gb:.2f} GB free.")
        else:
            self._safety_help_var.set(f"{base} No model loaded yet, so eligibility isn't judged on size.")

    def _on_safety_ratio_changed(self, _value: Optional[str] = None) -> None:
        self._update_safety_help_text()
        self._settings_set("gpu.safety_ratio", self.safety_ratio_var.get())
        self._recompute_and_apply()

    def _on_reserve_changed(self) -> None:
        self._settings_set("gpu.reserve_mb", self.vram_reserve_mb_var.get())
        self._recompute_and_apply()

    def reset_gpu_settings_to_auto(self) -> None:
        for state in self._gpu_states.values():
            state.enabled = True
            state.mode = "auto"
            state.manual_weight_pct = 0.0
        self.safety_ratio_var.set(DEFAULT_SAFETY_RATIO)
        self.vram_reserve_mb_var.set(DEFAULT_RESERVE_MB)
        self._settings_set("gpu.safety_ratio", DEFAULT_SAFETY_RATIO)
        self._settings_set("gpu.reserve_mb", DEFAULT_RESERVE_MB)
        self._update_safety_help_text()
        self._recompute_and_apply()

    # -- scrollable card list --------------------------------------------

    def _build_card_list(self, parent: tk.Widget) -> ttk.Frame:
        outer = ttk.Frame(parent)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self._card_container = ttk.Frame(canvas)
        self._card_container.columnconfigure(0, weight=1)

        self._card_container.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self._card_container, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(1, width=e.width),
        )

        canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")

        def _on_mousewheel(event):
            delta = -1 if (event.num == 5 or getattr(event, "delta", 0) < 0) else 1
            canvas.yview_scroll(-delta, "units")

        canvas.bind("<Enter>", lambda e: self._bind_mousewheel(canvas, _on_mousewheel))
        canvas.bind("<Leave>", lambda e: self._unbind_mousewheel(canvas))

        self._empty_state_label = ttk.Label(
            self._card_container,
            text="No GPUs detected yet. Click \u201cRescan GPUs\u201d.",
            foreground="#888",
        )
        self._empty_state_label.grid(row=0, column=0, padx=8, pady=16)

        return outer

    @staticmethod
    def _bind_mousewheel(widget: tk.Widget, handler: Callable) -> None:
        widget.bind_all("<MouseWheel>", handler)  # Windows / macOS
        widget.bind_all("<Button-4>", handler)    # Linux scroll up
        widget.bind_all("<Button-5>", handler)    # Linux scroll down

    @staticmethod
    def _unbind_mousewheel(widget: tk.Widget) -> None:
        widget.unbind_all("<MouseWheel>")
        widget.unbind_all("<Button-4>")
        widget.unbind_all("<Button-5>")

    def _rebuild_cards(self) -> None:
        for card in self._gpu_cards.values():
            card.frame.destroy()
        self._gpu_cards.clear()

        if not self._gpu_infos:
            self._empty_state_label.grid(row=0, column=0, padx=8, pady=16)
            return

        self._empty_state_label.grid_forget()

        for row, index in enumerate(sorted(self._gpu_infos)):
            card = _GPUCard(
                parent=self._card_container,
                index=index,
                on_enabled_toggle=self._on_card_enabled_toggle,
                on_mode_change=self._on_card_mode_change,
                on_weight_change=self._on_card_weight_change,
            )
            card.frame.grid(row=row, column=0, sticky="ew", pady=4)
            self._gpu_cards[index] = card

    def _refresh_card(self, index: int) -> None:
        card = self._gpu_cards.get(index)
        if card is not None:
            card.render(self._gpu_infos[index], self._gpu_states[index])

    # -- rescanning (threaded, non-blocking) -----------------------------

    def rescan_gpus(self) -> None:
        with self._gpu_scan_lock:
            if self._gpu_scan_in_progress:
                return
            self._gpu_scan_in_progress = True

        self._rescan_btn.state(["disabled"])
        self._gpu_status_var.set("Scanning for GPUs\u2026")

        manager = getattr(self, "gpu_manager", None)
        binary = getattr(self, "transcribe_binary", None)

        def worker():
            try:
                if manager is None or not binary:
                    raise RuntimeError("gpu_manager / transcribe_binary not configured.")
                infos = manager.enumerate_all(binary)
                self._gpu_scan_queue.put(("ok", infos))
            except Exception as exc:  # surface any failure to the UI; never crash the thread
                self._gpu_scan_queue.put(("error", exc))

        threading.Thread(target=worker, daemon=True, name="gpu-rescan").start()
        self._gpu_scan_poll_job = self.after(SCAN_POLL_INTERVAL_MS, self._poll_scan_queue)

    def _poll_scan_queue(self) -> None:
        try:
            status, payload = self._gpu_scan_queue.get_nowait()
        except queue.Empty:
            self._gpu_scan_poll_job = self.after(SCAN_POLL_INTERVAL_MS, self._poll_scan_queue)
            return

        self._gpu_scan_in_progress = False
        self._rescan_btn.state(["!disabled"])

        if status == "error":
            self._gpu_status_var.set(f"Scan failed: {payload}")
            return

        self._apply_scan_results(payload)

    def _apply_scan_results(self, infos: list[GPUInfo]) -> None:
        # Build the complete new dicts before touching self._gpu_infos /
        # self._gpu_states, so nothing downstream ever sees a partial
        # update -- see design note (1).
        new_infos = {info.index: info for info in infos}
        new_states: dict[int, GPUState] = {}
        for index in new_infos:
            existing = self._gpu_states.get(index)
            new_states[index] = existing if existing is not None else self._load_state_or_default(index)

        self._gpu_infos = new_infos
        self._gpu_states = new_states

        if not self._gpu_infos:
            self._gpu_status_var.set("No Vulkan-capable GPUs found.")
        else:
            enabled_count = sum(1 for s in self._gpu_states.values() if s.enabled)
            self._gpu_status_var.set(f"{len(self._gpu_infos)} GPU(s) found \u2014 {enabled_count} enabled.")

        self._rebuild_cards()
        self._recompute_and_apply()

    def _load_state_or_default(self, index: int) -> GPUState:
        saved_all = self._settings_get("gpu.device_states", {}) or {}
        saved = saved_all.get(str(index))
        state = GPUState()
        if saved:
            state.enabled = saved.get("enabled", state.enabled)
            state.mode = saved.get("mode", state.mode)
            state.manual_weight_pct = saved.get("manual_weight_pct", state.manual_weight_pct)
        return state

    # -- cheap periodic VRAM refresh (optional; needs gpu_manager.probe) --

    def _schedule_live_refresh(self) -> None:
        self._gpu_live_refresh_job = self.after(LIVE_REFRESH_INTERVAL_MS, self._run_live_refresh)

    def _run_live_refresh(self) -> None:
        manager = getattr(self, "gpu_manager", None)
        binary = getattr(self, "transcribe_binary", None)
        probe = getattr(manager, "probe", None) if manager is not None else None

        if not self._gpu_infos or not callable(probe) or not binary or self._gpu_scan_in_progress:
            self._schedule_live_refresh()
            return

        indices = list(self._gpu_infos)

        def worker():
            updates: dict[int, GPUInfo] = {}
            for index in indices:
                try:
                    updates[index] = probe(binary, index)
                except Exception:
                    continue  # one bad probe shouldn't drop the rest
            self._gpu_live_queue.put(updates)

        threading.Thread(target=worker, daemon=True, name="gpu-live-refresh").start()
        self._gpu_live_poll_job = self.after(SCAN_POLL_INTERVAL_MS, self._poll_live_refresh)

    def _poll_live_refresh(self) -> None:
        try:
            updates = self._gpu_live_queue.get_nowait()
        except queue.Empty:
            self._gpu_live_poll_job = self.after(SCAN_POLL_INTERVAL_MS, self._poll_live_refresh)
            return

        for index, info in updates.items():
            if index in self._gpu_infos:
                self._gpu_infos[index] = info

        if updates:
            self._recompute_and_apply()
        self._schedule_live_refresh()

    # -- card callbacks ----------------------------------------------------

    def _on_card_enabled_toggle(self, index: int, enabled: bool) -> None:
        self._gpu_states[index].enabled = enabled
        self._recompute_and_apply()

    def _on_card_mode_change(self, index: int, mode: str) -> None:
        state = self._gpu_states[index]
        state.mode = mode
        if mode == "manual" and state.manual_weight_pct == 0.0:
            # Seed with the last computed auto share so flipping to
            # Manual doesn't jump the allocation straight to zero.
            state.manual_weight_pct = state.computed_weight_pct
        self._recompute_and_apply()

    def _on_card_weight_change(self, index: int, value: float) -> None:
        state = self._gpu_states[index]
        if state.mode != "manual":
            return  # slider should be disabled in auto mode; guard anyway
        state.manual_weight_pct = max(0.0, min(100.0, value))
        self._recompute_and_apply()

    # -- recompute + propagate ---------------------------------------------

    def _recompute_and_apply(self) -> None:
        result = compute_weighted_chunks(
            infos=self._gpu_infos,
            states=self._gpu_states,
            safety_ratio=self.safety_ratio_var.get(),
            reserve_bytes=self.vram_reserve_mb_var.get() * MB,
            model_size_bytes=self._get_model_size_bytes(),
        )

        for index, values in result.items():
            state = self._gpu_states[index]
            state.eligible = values["eligible"]
            state.computed_weight_pct = values["weight_pct"]

        for index in self._gpu_infos:
            self._refresh_card(index)

        self._write_vulkan_devices_var()
        self._persist_gpu_states()

    def _write_vulkan_devices_var(self) -> None:
        """Serializes the active GPU selection into `self.vulkan_devices_var`
        (a tk.StringVar already read by command_builder.py / vulkan.py) and
        stashes per-GPU weights on `self.gpu_weight_overrides` for
        transcriber.py to use when sizing each worker's audio chunk.

        If the existing var expects a different format than a comma-
        separated index list, override `_serialize_device_list` --
        everything else in this file is agnostic to that choice.
        """
        active = sorted(
            index for index, state in self._gpu_states.items()
            if state.enabled and state.eligible and state.computed_weight_pct > 0
        )

        var = getattr(self, "vulkan_devices_var", None)
        if var is not None:
            var.set(self._serialize_device_list(active))

        self.gpu_weight_overrides = {
            index: self._gpu_states[index].computed_weight_pct for index in active
        }

    def _serialize_device_list(self, indices: list[int]) -> str:
        return ",".join(str(i) for i in indices)

    def _persist_gpu_states(self) -> None:
        payload = {
            str(index): {
                "enabled": state.enabled,
                "mode": state.mode,
                "manual_weight_pct": state.manual_weight_pct,
            }
            for index, state in self._gpu_states.items()
        }
        self._settings_set("gpu.device_states", payload)

    # -- misc public helpers -------------------------------------------

    def get_active_gpu_summary(self) -> str:
        """One-line human-readable summary, e.g. "GPU 0 (63%), GPU 2 (37%)" --
        handy for a status bar elsewhere in the app."""
        parts = [
            f"GPU {index} ({state.computed_weight_pct:.0f}%)"
            for index, state in sorted(self._gpu_states.items())
            if state.enabled and state.eligible and state.computed_weight_pct > 0
        ]
        return ", ".join(parts) if parts else "No GPUs selected"


# ---------------------------------------------------------------------------
# Standalone demo -- run this file directly to see the panel with fake
# GPUs. Not used by the real app; swap _FakeGPUManager for the real
# app.backends.gpu_manager.GPUManager.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import time

    class _FakeGPUManager:
        def enumerate_all(self, binary):
            time.sleep(0.6)  # simulate subprocess latency
            return [
                GPUInfo(0, "AMD Radeon RX 7900 XTX", 24 * GB, int(24 * GB * random.uniform(0.3, 0.9))),
                GPUInfo(1, "NVIDIA RTX 4070", 12 * GB, int(12 * GB * random.uniform(0.1, 0.6))),
                GPUInfo(2, "Intel Arc A770", 16 * GB, int(16 * GB * random.uniform(0.4, 0.95))),
            ]

        def probe(self, binary, index):
            time.sleep(0.05)
            totals = {0: 24 * GB, 1: 12 * GB, 2: 16 * GB}
            names = {0: "AMD Radeon RX 7900 XTX", 1: "NVIDIA RTX 4070", 2: "Intel Arc A770"}
            total = totals[index]
            return GPUInfo(index, names[index], total, int(total * random.uniform(0.1, 0.95)))

    class _FakeSettings:
        def __init__(self):
            self._data = {}

        def get(self, key, default=None):
            return self._data.get(key, default)

        def set(self, key, value):
            self._data[key] = value

    class DemoWindow(GPUControlPanelMixin, tk.Tk):
        def __init__(self):
            tk.Tk.__init__(self)
            self.title("HandyAidGUI \u2014 GPU Panel Demo")
            self.geometry("680x520")

            self.gpu_manager = _FakeGPUManager()
            self.transcribe_binary = "/usr/local/bin/transcribe-cli"
            self.settings = _FakeSettings()
            self.vulkan_devices_var = tk.StringVar()
            self._demo_model_bytes = 6 * GB

            notebook = ttk.Notebook(self)
            notebook.pack(fill="both", expand=True)
            notebook.add(self.build_gpu_tab(notebook), text="GPU")

            self.vulkan_devices_var.trace_add(
                "write", lambda *_: print("vulkan_devices_var ->", self.vulkan_devices_var.get())
            )

        def get_model_size_bytes(self):
            return self._demo_model_bytes

    DemoWindow().mainloop()
