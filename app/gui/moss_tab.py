# app/gui/moss_tab.py

import os
import torch
import librosa
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QFileDialog, 
                               QSpinBox, QTextEdit, QProgressBar)
from PySide6.QtCore import QThread, Signal

class MossTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.model = None  # will be loaded on first use

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.select_btn = QPushButton("Select Audio")
        self.select_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_btn)
        layout.addLayout(file_layout)

        # Chunk size
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(QLabel("Chunk duration (seconds):"))
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(10, 600)
        self.chunk_spin.setValue(60)
        chunk_layout.addWidget(self.chunk_spin)
        layout.addLayout(chunk_layout)

        # Progress bar
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # Process button
        self.process_btn = QPushButton("Transcribe")
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)

        # Output text area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.m4a)"
        )
        if path:
            self.file_label.setText(path)
            self.file_path = path

    def start_processing(self):
        if not hasattr(self, 'file_path') or not self.file_path:
            self.output_text.append("Please select an audio file first.")
            return
        # Disable UI during processing
        self.process_btn.setEnabled(False)
        self.progress.setValue(0)
        self.output_text.clear()

        # Run processing in a separate thread to keep UI responsive
        self.worker = ProcessingWorker(self.file_path, self.chunk_spin.value())
        self.worker.progress_updated.connect(self.progress.setValue)
        self.worker.log_message.connect(self.output_text.append)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()

    def on_processing_finished(self, transcript):
        self.output_text.append("\n--- Final Transcript ---\n")
        self.output_text.append(transcript)
        self.process_btn.setEnabled(True)
