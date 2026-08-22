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
# app/gui/moss_tab.py (continued)

from PySide6.QtCore import QThread, Signal
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq  # adjust to your MOSS wrapper

class ProcessingWorker(QThread):
    progress_updated = Signal(int)
    log_message = Signal(str)

    def __init__(self, file_path, chunk_duration):
        super().__init__()
        self.file_path = file_path
        self.chunk_duration = chunk_duration

    def run(self):
        try:
            self.log_message.emit("Loading audio...")
            audio, sr = librosa.load(self.file_path, sr=16000, mono=True)
            total_samples = len(audio)
            chunk_samples = int(self.chunk_duration * sr)
            num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

            # Load MOSS model (lazy load, once)
            model, processor = self.load_model()

            all_segments = []          # will hold stitched segments
            speaker_map = {}           # maps local speaker IDs to global IDs
            next_global_id = 1

            for i in range(num_chunks):
                start = i * chunk_samples
                end = min(start + chunk_samples, total_samples)
                chunk_audio = audio[start:end]

                # Process chunk with MOSS
                self.log_message.emit(f"Processing chunk {i+1}/{num_chunks}...")
                result = self.transcribe_chunk(model, processor, chunk_audio, sr)

                # Offset timestamps
                offset = start / sr
                for seg in result:
                    seg['start'] += offset
                    seg['end'] += offset

                # Remap speaker labels (simple: each new local ID gets a new global ID)
                for seg in result:
                    local_id = seg.get('speaker', 'S01')
                    if local_id not in speaker_map:
                        global_id = f"G{next_global_id:02d}"
                        speaker_map[local_id] = global_id
                        next_global_id += 1
                    seg['speaker'] = speaker_map[local_id]

                all_segments.extend(result)

                # Update progress
                progress = int((i + 1) / num_chunks * 100)
                self.progress_updated.emit(progress)

                # Clear GPU cache after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Sort all segments by start time
            all_segments.sort(key=lambda x: x['start'])

            # Format final transcript
            transcript = self.format_transcript(all_segments)
            self.log_message.emit("Processing complete.")
            self.finished.emit(transcript)

        except Exception as e:
            self.log_message.emit(f"Error: {str(e)}")
            self.finished.emit("")

    finished = Signal(str)

    def load_model(self):
        # Adjust this to your actual MOSS loading code.
        # Assuming you have a function in your backends that returns model and processor.
        from app.backends.moss_backend import load_moss_model  # adapt path
        return load_moss_model()

    def transcribe_chunk(self, model, processor, audio, sr):
        # This is a placeholder – replace with actual MOSS inference.
        # The model should return a list of dicts with keys: start, end, speaker, text.
        # For demonstration, we simulate a simple result.
        # In real use, you'd call model.generate(...) and parse the output.
        # Here we mimic a dummy transcription for one segment.
        dummy = [
            {"start": 0.0, "end": len(audio)/sr, "speaker": "S01", "text": "Sample transcription for this chunk."}
        ]
        return dummy

    def format_transcript(self, segments):
        lines = []
        for seg in segments:
            speaker = seg.get('speaker', 'Unknown')
            text = seg.get('text', '').strip()
            if text:
                lines.append(f"[{speaker}] {text}")
        return "\n".join(lines)
