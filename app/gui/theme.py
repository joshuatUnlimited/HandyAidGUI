import tkinter as tk
from tkinter import ttk


class ThemeMixin:
    BG = "#0b1120"
    PANEL = "#111b2e"
    PANEL_2 = "#18253a"
    BORDER = "#4b6282"
    TEXT = "#ffffff"
    MUTED = "#c0cad8"
    ACCENT = "#66b3ff"
    SUCCESS = "#4ade80"
    WARNING = "#facc15"
    DANGER = "#fb7185"

    def setup_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("Card.TFrame", background=self.PANEL_2)

        style.configure(
            "TLabel",
            background=self.BG,
            foreground=self.TEXT,
        )
        style.configure(
            "Muted.TLabel",
            background=self.BG,
            foreground=self.MUTED,
        )
        style.configure(
            "Panel.TLabel",
            background=self.PANEL,
            foreground=self.TEXT,
        )
        style.configure(
            "Card.TLabel",
            background=self.PANEL_2,
            foreground=self.TEXT,
        )
        style.configure(
            "Title.TLabel",
            background=self.BG,
            foreground=self.TEXT,
            font=("Segoe UI", 22, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BG,
            foreground=self.MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Metric.TLabel",
            background=self.PANEL_2,
            foreground=self.TEXT,
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "MetricCaption.TLabel",
            background=self.PANEL_2,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )

        style.configure(
            "TEntry",
            fieldbackground="#0f172a",
            foreground=self.TEXT,
            insertcolor=self.TEXT,
            bordercolor=self.BORDER,
        )
        style.map(
            "TEntry",
            fieldbackground=[("disabled", "#111827")],
        )

        style.configure(
            "TCombobox",
            fieldbackground="#0f172a",
            background="#0f172a",
            foreground=self.TEXT,
            arrowcolor=self.TEXT,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#0f172a")],
            foreground=[("readonly", self.TEXT)],
        )

        style.configure(
            "TButton",
            background="#2b4566",
            foreground="#ffffff",
            bordercolor="#58739a",
            padding=(13, 8),
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "TButton",
            background=[
                ("active", "#3a5f8c"),
                ("pressed", "#23405f"),
                ("disabled", "#243248"),
            ],
            foreground=[("disabled", "#71809a")],
        )

        style.configure(
            "Accent.TButton",
            background=self.ACCENT,
            foreground="#08111f",
            bordercolor=self.ACCENT,
            padding=(15, 9),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[
                ("active", "#93c5fd"),
                ("disabled", "#475569"),
            ],
        )

        style.configure(
            "Danger.TButton",
            background="#7f1d1d",
            foreground=self.TEXT,
            bordercolor="#991b1b",
            padding=(15, 9),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#991b1b")],
        )

        style.configure(
            "TNotebook",
            background=self.BG,
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            background="#1a2940",
            foreground="#dbe7f5",
            padding=(15, 9),
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", "#203653"),
                ("active", "#274263"),
            ],
            foreground=[
                ("selected", "#ffffff"),
                ("active", "#ffffff"),
            ],
        )

        style.configure(
            "TProgressbar",
            troughcolor="#0b1220",
            background=self.ACCENT,
            lightcolor=self.ACCENT,
            darkcolor=self.ACCENT,
            borderwidth=0,
        )

        style.configure(
            "Horizontal.TScale",
            background=self.PANEL,
        )

        style.configure(
            "TRadiobutton",
            background=self.PANEL,
            foreground=self.TEXT,
        )
        style.map(
            "TRadiobutton",
            background=[("active", self.PANEL)],
            foreground=[("active", self.TEXT)],
        )

        style.configure(
            "TCheckbutton",
            background=self.PANEL,
            foreground=self.TEXT,
        )
        style.map(
            "TCheckbutton",
            background=[("active", self.PANEL)],
            foreground=[("active", self.TEXT)],
        )

        style.configure(
            "Status.TLabel",
            background="#0a1324",
            foreground="#dbeafe",
            padding=(8, 5),
            font=("Segoe UI", 9),
        )

    def configure_text_widget(self, widget, bg=None):
        """
        Configure a Tkinter Text widget.

        `bg` is optional. The previous implementation referenced an undefined
        local variable named `bg`, causing startup to fail as soon as the
        transcript panel was constructed.
        """
        widget.configure(
            bg=bg or "#070e1a",
            fg="#ffffff",
            insertbackground=self.TEXT,
            selectbackground="#2f6fb0",
            selectforeground="#ffffff",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#5b7496",
            highlightcolor="#8ac7ff",
            font=("Cascadia Mono", 10),
        )
