"""Tests for the Google Gemini TTS provider in tools/tts_tool.py."""

import base64
import struct
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_BASE_URL",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_pcm_bytes():
    # 0.1s of silence at 24kHz mono 16-bit = 4800 bytes
    return b"\x00" * 4800


@pytest.fixture
def mock_gemini_response(fake_pcm_bytes):
    """A successful Gemini generateContent response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/L16;codec=pcm;rate=24000",
                                "data": base64.b64encode(fake_pcm_bytes).decode(),
                            }
                        }
                    ]
                }
            }
        ]
    }
    return resp


class TestWrapPcmAsWav:
    def test_riff_header_structure(self):
        from tools.tts_tool import _wrap_pcm_as_wav

        pcm = b"\x01\x02\x03\x04" * 10
        wav = _wrap_pcm_as_wav(pcm, sample_rate=24000, channels=1, sample_width=2)

        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        # Audio format (PCM=1)
        assert struct.unpack("<H", wav[20:22])[0] == 1
        # Channels
        assert struct.unpack("<H", wav[22:24])[0] == 1
        # Sample rate
        assert struct.unpack("<I", wav[24:28])[0] == 24000
        # Bits per sample
        assert struct.unpack("<H", wav[34:36])[0] == 16
        assert wav[36:40] == b"data"
        assert wav[44:] == pcm

    def test_header_size_is_44(self):
        from tools.tts_tool import _wrap_pcm_as_wav

        pcm = b"\xff" * 100
        wav = _wrap_pcm_as_wav(pcm)
        assert len(wav) == 44 + len(pcm)


class TestGenerateGeminiTts:
    def test_missing_api_key_raises_value_error(self, tmp_path):
        from tools.tts_tool import _generate_gemini_tts

        output_path = str(tmp_path / "test.wav")
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            _generate_gemini_tts("Hello", output_path, {})

    def test_google_api_key_fallback(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GOOGLE_API_KEY", "from-google-env")
        output_path = str(tmp_path / "test.wav")

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", output_path, {})

        # Confirm it used the GOOGLE_API_KEY as the query parameter
        _, kwargs = mock_post.call_args
        assert kwargs["params"]["key"] == "from-google-env"

    def test_wav_output_fast_path(self, tmp_path, monkeypatch, mock_gemini_response, fake_pcm_bytes):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        output_path = str(tmp_path / "test.wav")

        with patch("requests.post", return_value=mock_gemini_response):
            result = _generate_gemini_tts("Hi", output_path, {})

        assert result == output_path
        data = (tmp_path / "test.wav").read_bytes()
        assert data[:4] == b"RIFF"
        assert data[8:12] == b"WAVE"
        # Audio payload should match the PCM we put in
        assert data[44:] == fake_pcm_bytes

    def test_x_goog_api_client_header_is_set(self, tmp_path, monkeypatch, mock_gemini_response):
        """Gemini TTS requests should include Hermes client context."""
        from hermes_cli import __version__
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), {})

        headers = mock_post.call_args[1]["headers"]
        assert headers["X-Goog-Api-Client"] == f"hermes-agent/{__version__}"

    def test_default_voice_and_model(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import (
            DEFAULT_GEMINI_TTS_MODEL,
            DEFAULT_GEMINI_TTS_VOICE,
            _generate_gemini_tts,
        )

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), {})

        args, kwargs = mock_post.call_args
        assert DEFAULT_GEMINI_TTS_MODEL in args[0]
        payload = kwargs["json"]
        voice = (
            payload["generationConfig"]["speechConfig"]["voiceConfig"]
            ["prebuiltVoiceConfig"]["voiceName"]
        )
        assert voice == DEFAULT_GEMINI_TTS_VOICE

    def test_custom_voice(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = {"gemini": {"voice": "Puck"}}

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), config)

        payload = mock_post.call_args[1]["json"]
        voice = (
            payload["generationConfig"]["speechConfig"]["voiceConfig"]
            ["prebuiltVoiceConfig"]["voiceName"]
        )
        assert voice == "Puck"


    def test_audio_tag_rewrite_failure_falls_back_to_original_text(
        self, tmp_path, monkeypatch, mock_gemini_response, caplog
    ):
        from tools.tts_tool import _generate_gemini_tts

        config = {
            "gemini": {
                "model": "gemini-3.1-flash-tts-preview",
                "audio_tags": True,
            }
        }
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("boom")), \
             patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi there.", str(tmp_path / "test.wav"), config)

        prompt_text = mock_post.call_args[1]["json"]["contents"][0]["parts"][0]["text"]
        assert prompt_text == "Hi there."
        assert "audio tag rewrite failed" in caplog.text


class TestGeminiInCheckRequirements:
    def test_gemini_api_key_satisfies_requirements(self, monkeypatch):
        from tools.tts_tool import check_tts_requirements

        # Strip everything else
        for key in (
            "ELEVENLABS_API_KEY",
            "OPENAI_API_KEY",
            "VOICE_TOOLS_OPENAI_KEY",
            "MINIMAX_API_KEY",
            "XAI_API_KEY",
            "MISTRAL_API_KEY",
            "GOOGLE_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "k")

        # Force edge_tts import to fail so we actually hit the gemini check
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "edge_tts":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with patch(
            "tools.tts_tool._load_tts_config",
            return_value={"provider": "gemini"},
        ), patch("builtins.__import__", side_effect=fake_import):
            assert check_tts_requirements() is True


def _gemini_response_with(audio_bytes):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"inlineData": {
            "mimeType": "audio/wav",
            "data": base64.b64encode(audio_bytes).decode(),
        }}]}}]
    }
    return resp


class TestGeminiVoiceFidelity:
    """The live voice must match the sample it was tuned on (LINE 晨報 voice, 2026-10-09)."""

    def test_complete_wav_from_gemini_is_not_wrapped_again(self, tmp_path, monkeypatch, fake_pcm_bytes):
        from tools.tts_tool import _generate_gemini_tts, _wrap_pcm_as_wav

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        # gemini-3.8-flash-tts returns a full WAV with a trailing C2PA chunk.
        full_wav = _wrap_pcm_as_wav(fake_pcm_bytes) + b"C2PA" + struct.pack("<I", 8) + b"\xab" * 8

        with patch("requests.post", return_value=_gemini_response_with(full_wav)):
            _generate_gemini_tts("Hi", str(tmp_path / "t.wav"), {})

        data = (tmp_path / "t.wav").read_bytes()
        assert data == full_wav
        assert data.count(b"RIFF") == 1

    def test_mp3_conversion_uses_128k(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        done = SimpleNamespace(returncode=0, stderr=b"")
        with patch("requests.post", return_value=mock_gemini_response), \
                patch("tools.tts_tool.shutil.which", return_value="/usr/bin/ffmpeg"), \
                patch("tools.tts_tool.subprocess.run", return_value=done) as run:
            _generate_gemini_tts("Hi", str(tmp_path / "t.mp3"), {})

        cmd = run.call_args[0][0]
        assert cmd[cmd.index("-b:a") + 1] == "128k"

    def test_persona_preamble_false_sends_persona_exactly(self):
        from tools.tts_tool import _compose_gemini_tts_prompt

        persona = "# AUDIO PROFILE: X\n### DIRECTOR'S NOTES\nStyle: soft.\n#### TRANSCRIPT\n{{transcript}}"
        out = _compose_gemini_tts_prompt("小灰早安", {"persona_preamble": False}, persona_prompt=persona)
        assert out == "# AUDIO PROFILE: X\n### DIRECTOR'S NOTES\nStyle: soft.\n#### TRANSCRIPT\n小灰早安"

    def test_preamble_kept_by_default(self):
        from tools.tts_tool import _compose_gemini_tts_prompt

        out = _compose_gemini_tts_prompt("hi", {}, persona_prompt="Style: soft.")
        assert out.startswith("Synthesize speech from the TRANSCRIPT only.")
        assert out.endswith("Style: soft.\n\n#### TRANSCRIPT\nhi")

    def test_backslash_in_transcript_is_kept_literally(self):
        from tools.tts_tool import _compose_gemini_tts_prompt

        text = r"路徑 C:\new\1 \g<0>"
        out = _compose_gemini_tts_prompt(text, {"persona_preamble": False}, persona_prompt="A\n{{transcript}}")
        assert out == "A\n" + text


def _tone_wav(f0_start, f0_end, tilt, seconds=2.0, sr=24000):
    """Harmonic tone gliding f0_start -> f0_end Hz; harmonics fall off as 1/k**tilt (higher tilt = darker)."""
    np = pytest.importorskip("numpy")
    from tools.tts_tool import _wrap_pcm_as_wav

    t = np.arange(int(sr * seconds))
    phase = 2 * np.pi * np.cumsum(np.linspace(f0_start, f0_end, t.size)) / sr
    x = sum(np.sin(k * phase) / k ** tilt for k in range(1, 30))
    x = 0.3 * x / np.abs(x).max()
    return _wrap_pcm_as_wav((x * 32767).astype("<i2").tobytes(), sample_rate=sr)


class TestGeminiTakes:
    """tts.gemini.takes: retake flat/bright performances, keep the one closest to the tuned voice."""

    def _takes(self, *wavs):
        calls = []

        def take():
            calls.append(1)
            w = wavs[len(calls) - 1]
            if isinstance(w, Exception):
                raise w
            return w

        return take, calls

    def test_features_tell_lively_from_flat_and_dark_from_bright(self):
        from tools.tts_tool import _gemini_take_features

        lively = _gemini_take_features(_tone_wav(130, 300, tilt=1.5))
        flat = _gemini_take_features(_tone_wav(180, 185, tilt=1.5))
        bright = _gemini_take_features(_tone_wav(180, 185, tilt=0.5))
        assert lively["pitch_range_hz"] >= 100 > flat["pitch_range_hz"]
        assert lively["brightness_hz"] <= 1450 < bright["brightness_hz"]

    def test_single_request_by_default(self):
        from tools.tts_tool import _pick_gemini_take

        flat = _tone_wav(180, 185, tilt=1.5)
        take, calls = self._takes(flat, flat, flat)
        assert _pick_gemini_take(take, {}) == flat
        assert len(calls) == 1

    def test_first_take_kept_when_it_matches(self):
        from tools.tts_tool import _pick_gemini_take

        lively = _tone_wav(130, 300, tilt=1.5)
        take, calls = self._takes(lively, lively, lively)
        assert _pick_gemini_take(take, {"takes": 3}) == lively
        assert len(calls) == 1

    def test_retakes_until_one_matches(self):
        from tools.tts_tool import _pick_gemini_take

        flat, lively = _tone_wav(180, 185, tilt=1.5), _tone_wav(130, 300, tilt=1.5)
        take, calls = self._takes(flat, lively, flat)
        assert _pick_gemini_take(take, {"takes": 3}) == lively
        assert len(calls) == 2

    def test_keeps_closest_when_none_match(self):
        from tools.tts_tool import _pick_gemini_take

        flat, flat_bright = _tone_wav(180, 185, tilt=1.5), _tone_wav(180, 185, tilt=0.5)
        take, calls = self._takes(flat_bright, flat, flat_bright)
        assert _pick_gemini_take(take, {"takes": 3}) == flat
        assert len(calls) == 3

    def test_failed_retake_keeps_best_so_far(self):
        from tools.tts_tool import _pick_gemini_take

        flat = _tone_wav(180, 185, tilt=1.5)
        take, calls = self._takes(flat, RuntimeError("Gemini TTS API error (HTTP 503)"))
        assert _pick_gemini_take(take, {"takes": 3}) == flat
        assert len(calls) == 2

    def test_bad_takes_values_mean_a_single_request(self):
        from tools.tts_tool import _pick_gemini_take

        flat = _tone_wav(180, 185, tilt=1.5)
        for bad in ("abc", 0, -2, None, float("inf")):
            take, calls = self._takes(flat, flat, flat)
            assert _pick_gemini_take(take, {"takes": bad}) == flat
            assert len(calls) == 1, bad

    def test_takes_capped_at_five(self):
        from tools.tts_tool import _pick_gemini_take

        flat = _tone_wav(180, 185, tilt=1.5)
        take, calls = self._takes(*([flat] * 50))
        _pick_gemini_take(take, {"takes": 50})
        assert len(calls) == 5

    def test_bad_thresholds_fall_back_to_defaults(self):
        from tools.tts_tool import _pick_gemini_take

        lively = _tone_wav(130, 300, tilt=1.5)
        take, calls = self._takes(lively, lively, lively)
        cfg = {"takes": 3, "take_min_pitch_range_hz": 0, "take_max_brightness_hz": "loud"}
        assert _pick_gemini_take(take, cfg) == lively
        assert len(calls) == 1

    def test_unmeasurable_first_take_is_used_without_retakes(self):
        from tools.tts_tool import _pick_gemini_take, _wrap_pcm_as_wav

        silence = _wrap_pcm_as_wav(b"\x00\x00" * 48000)
        odd_bytes = _wrap_pcm_as_wav(b"\x01" * 48001)
        for first in (silence, odd_bytes, b"not a wav"):
            take, calls = self._takes(first, first, first)
            assert _pick_gemini_take(take, {"takes": 3}) == first
            assert len(calls) == 1

    def test_malformed_retake_keeps_best_so_far(self):
        from tools.tts_tool import _pick_gemini_take

        flat = _tone_wav(180, 185, tilt=1.5)
        take, calls = self._takes(flat, ValueError("Expecting value: line 1 column 1"))
        assert _pick_gemini_take(take, {"takes": 3}) == flat
        assert len(calls) == 2

    def test_without_numpy_only_one_request(self):
        import importlib.util

        from tools.tts_tool import _pick_gemini_take

        real_find_spec = importlib.util.find_spec
        flat = _tone_wav(180, 185, tilt=1.5)
        take, calls = self._takes(flat, flat, flat)
        with patch("importlib.util.find_spec", side_effect=lambda name, *a: None if name == "numpy" else real_find_spec(name, *a)):
            assert _pick_gemini_take(take, {"takes": 3}) == flat
        assert len(calls) == 1

    def test_generate_gemini_tts_retakes_through_the_api(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        flat, lively = _tone_wav(180, 185, tilt=1.5), _tone_wav(130, 300, tilt=1.5)
        responses = [_gemini_response_with(flat), _gemini_response_with(lively)]
        with patch("requests.post", side_effect=responses) as post:
            _generate_gemini_tts("Hi", str(tmp_path / "t.wav"), {"gemini": {"takes": 3}})
        assert post.call_count == 2
        assert (tmp_path / "t.wav").read_bytes() == lively

    def test_malformed_header_first_take_is_used_without_retakes(self):
        from tools.tts_tool import _pick_gemini_take

        bad = bytearray(_tone_wav(180, 185, tilt=1.5))
        bad[16] = 100  # fmt chunk size: the stdlib wave reader raises a bare RuntimeError
        bad = bytes(bad)
        take, calls = self._takes(bad, bad, bad)
        assert _pick_gemini_take(take, {"takes": 3}) == bad
        assert len(calls) == 1

    def test_retake_connection_error_does_not_log_the_key(self, caplog):
        import logging

        import requests

        from tools.tts_tool import _pick_gemini_take

        flat = _tone_wav(180, 185, tilt=1.5)
        leak = requests.ConnectionError("HTTPSConnectionPool: /models/m:generateContent?key=SECRET-KEY-123")
        take, calls = self._takes(flat, leak)
        with caplog.at_level(logging.INFO):
            assert _pick_gemini_take(take, {"takes": 3}) == flat
        assert len(calls) == 2
        assert "SECRET-KEY-123" not in caplog.text
        assert "ConnectionError" in caplog.text

    def test_mistyped_setting_is_logged(self, caplog):
        import logging

        from tools.tts_tool import _pick_gemini_take

        flat = _tone_wav(180, 185, tilt=1.5)
        take, calls = self._takes(flat)
        with caplog.at_level(logging.WARNING):
            _pick_gemini_take(take, {"takes": "three"})
        assert "tts.gemini.takes='three'" in caplog.text
