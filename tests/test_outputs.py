from app.output.writers import OutputWriterMixin

def test_srt_time():
    assert OutputWriterMixin.srt_time(3661.25) == "01:01:01,250"

def test_markdown_basic():
    class X(OutputWriterMixin):
        audio_path_var = type("V", (), {"get": lambda s: "test.wav", "strip": lambda s: "test.wav"})()
        model_path_var = type("V", (), {"get": lambda s: "model.gguf", "strip": lambda s: "model.gguf"})()
        speaker_mode_var = type("V", (), {"get": lambda s: "multi"})()
        language_var = type("V", (), {"get": lambda s: "English (en)", "strip": lambda s: "English (en)"})()
        current_audio_duration = 10
        def format_seconds(self, x, always_hours=False):
            return "00:00:10"
    out = X().to_markdown("hello", [{"start":0,"end":2,"speaker":"S1","text":"hello"}])
    assert "**S1**" in out
