"""MuScripter Flash: low-latency streaming audio -> MIDI.

Flash is deliberately separate from the normal MuScriptor transformer path.
The published transformer is trained/evaluated on five-second segments, which
is excellent for transcription quality but is the wrong latency shape for a
live instrument. Flash instead keeps a short rolling analysis window and emits
note state changes immediately. A later high-accuracy MuScriptor pass can still
be used to correct/save the performance.

The first backend is a lightweight spectral detector. It is intentionally
small, deterministic, and CPU friendly so the streaming/MIDI architecture can
be exercised before a dedicated causal neural Flash model is trained.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FlashConfig:
    """Runtime tuning for the low-latency Flash engine."""

    sample_rate: int = 16_000
    window_ms: float = 128.0
    hop_ms: float = 32.0
    latency_budget_ms: float = 250.0
    min_midi: int = 21
    max_midi: int = 108
    max_polyphony: int = 12
    min_confidence: float = 0.46
    relative_threshold: float = 0.42
    silence_rms: float = 1e-4
    attack_frames: int = 1
    release_frames: int = 3
    velocity_floor: int = 32
    velocity_ceiling: int = 118

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 < self.hop_ms <= self.window_ms:
            raise ValueError("hop_ms must be > 0 and <= window_ms")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")
        if not 0 <= self.min_midi <= self.max_midi <= 127:
            raise ValueError("MIDI range must satisfy 0 <= min <= max <= 127")
        if self.max_polyphony <= 0:
            raise ValueError("max_polyphony must be positive")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0.0 <= self.relative_threshold <= 1.0:
            raise ValueError("relative_threshold must be in [0, 1]")
        if self.attack_frames <= 0 or self.release_frames <= 0:
            raise ValueError("attack_frames and release_frames must be positive")
        if not 1 <= self.velocity_floor <= self.velocity_ceiling <= 127:
            raise ValueError("velocity range must be inside [1, 127]")

    @property
    def window_samples(self) -> int:
        return max(1, round(self.sample_rate * self.window_ms / 1000.0))

    @property
    def hop_samples(self) -> int:
        return max(1, round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def nominal_note_on_latency_ms(self) -> float:
        """Worst-case algorithmic wait before compute time is added."""
        return self.window_ms + (self.attack_frames - 1) * self.hop_ms


@dataclass(frozen=True)
class PitchEstimate:
    midi_note: int
    confidence: float
    strength: float


@dataclass(frozen=True)
class FlashMidiEvent:
    type: Literal["note_on", "note_off"]
    note: int
    velocity: int
    stream_time: float
    confidence: float
    latency_ms: float


@dataclass
class FlashStats:
    analyses: int = 0
    note_ons: int = 0
    note_offs: int = 0
    max_compute_ms: float = 0.0
    last_compute_ms: float = 0.0
    budget_misses: int = 0


class SpectralPitchDetector:
    """Fast 88-key-ish polyphonic detector used by the Flash MVP.

    The detector scores each MIDI fundamental plus a few harmonics. A strong
    fundamental is weighted more heavily than its harmonics, which suppresses
    the common failure mode where a harmonic is mistaken for a new note while
    still allowing actual chords to produce multiple simultaneous candidates.
    """

    _HARMONIC_WEIGHTS = np.asarray([1.0, 0.42, 0.24, 0.16], dtype=np.float32)

    def __init__(self, config: FlashConfig):
        self.config = config
        self._window = np.hanning(config.window_samples).astype(np.float32)

        # Zero-padding improves frequency interpolation without adding latency.
        target_fft = max(2048, config.window_samples * 4)
        self._n_fft = 1 << (target_fft - 1).bit_length()
        self._bin_hz = config.sample_rate / self._n_fft

        self._notes = np.arange(config.min_midi, config.max_midi + 1, dtype=np.int16)
        freqs = 440.0 * np.power(2.0, (self._notes.astype(np.float64) - 69.0) / 12.0)

        harmonic_bins = np.full((len(self._notes), 4), -1, dtype=np.int32)
        for i, fundamental in enumerate(freqs):
            for h in range(1, 5):
                freq = fundamental * h
                if freq < config.sample_rate / 2:
                    harmonic_bins[i, h - 1] = round(freq / self._bin_hz)
        self._harmonic_bins = harmonic_bins

    @staticmethod
    def _local_peak(spectrum: np.ndarray, center: int) -> float:
        if center < 0 or center >= spectrum.size:
            return 0.0
        lo = max(0, center - 1)
        hi = min(spectrum.size, center + 2)
        return float(np.max(spectrum[lo:hi]))

    def detect(self, frame: np.ndarray) -> dict[int, PitchEstimate]:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size != self.config.window_samples:
            raise ValueError(
                f"expected {self.config.window_samples} samples, got {frame.size}"
            )

        centered = frame - float(np.mean(frame))
        rms = float(np.sqrt(np.mean(centered * centered) + 1e-12))
        if rms < self.config.silence_rms:
            return {}

        spectrum = np.abs(np.fft.rfft(centered * self._window, n=self._n_fft)).astype(
            np.float32
        )
        peak = float(np.max(spectrum))
        if peak <= 1e-12:
            return {}
        spectrum /= peak

        scores = np.zeros(len(self._notes), dtype=np.float32)
        for i, bins in enumerate(self._harmonic_bins):
            values: list[float] = []
            weights: list[float] = []
            for harmonic_index, center in enumerate(bins):
                if center < 0:
                    continue
                values.append(self._local_peak(spectrum, int(center)))
                weights.append(float(self._HARMONIC_WEIGHTS[harmonic_index]))

            if not values:
                continue

            fundamental = values[0]
            if len(values) > 1:
                harmonic_weights = np.asarray(weights[1:], dtype=np.float32)
                harmonic_values = np.asarray(values[1:], dtype=np.float32)
                harmonics = float(
                    np.dot(harmonic_values, harmonic_weights)
                    / max(float(np.sum(harmonic_weights)), 1e-9)
                )
            else:
                harmonics = 0.0

            scores[i] = 0.70 * fundamental + 0.30 * harmonics

        max_score = float(np.max(scores))
        if max_score <= 1e-6:
            return {}

        # Keep an absolute floor as well as a threshold relative to the current
        # strongest pitch. This stops low-level leakage/harmonics from filling
        # all available polyphony slots.
        threshold = max(
            self.config.min_confidence,
            max_score * self.config.relative_threshold,
        )
        indices = np.flatnonzero(scores >= threshold)
        if indices.size == 0:
            return {}

        ranked = indices[np.argsort(scores[indices])[::-1]][: self.config.max_polyphony]
        estimates: dict[int, PitchEstimate] = {}
        for index in ranked:
            score = float(scores[index])
            # Confidence is relative to the strongest pitch in this frame while
            # strength preserves the absolute harmonic score.
            confidence = min(1.0, score / max_score)
            note = int(self._notes[index])
            estimates[note] = PitchEstimate(
                midi_note=note,
                confidence=confidence,
                strength=min(1.0, score),
            )
        return estimates


@dataclass
class _PendingNote:
    frames: int = 0
    confidence: float = 0.0
    strength: float = 0.0


@dataclass
class _ActiveNote:
    velocity: int
    confidence: float
    missing_frames: int = 0


class FlashEngine:
    """Streaming note-state machine.

    Feed arbitrary mono PCM blocks with :meth:`process_block`. The engine owns
    the rolling window, analysis cadence, attack/release debouncing, and event
    timestamps. It never blocks waiting for a future five-second segment.
    """

    def __init__(
        self,
        config: FlashConfig | None = None,
        detector: SpectralPitchDetector | None = None,
    ):
        self.config = config or FlashConfig()
        self.detector = detector or SpectralPitchDetector(self.config)
        self.stats = FlashStats()

        self._ring = np.zeros(self.config.window_samples, dtype=np.float32)
        self._ring_filled = 0
        self._input = np.empty(0, dtype=np.float32)
        self._samples_consumed = 0
        self._pending: dict[int, _PendingNote] = {}
        self._active: dict[int, _ActiveNote] = {}

    @property
    def active_notes(self) -> frozenset[int]:
        return frozenset(self._active)

    def reset(self) -> None:
        self._ring.fill(0)
        self._ring_filled = 0
        self._input = np.empty(0, dtype=np.float32)
        self._samples_consumed = 0
        self._pending.clear()
        self._active.clear()
        self.stats = FlashStats()

    def _velocity(self, estimate: PitchEstimate) -> int:
        # Strength is intentionally mixed in so the loudest note is not always
        # velocity 127 merely because confidence is frame-relative.
        normalized = min(1.0, 0.55 * estimate.confidence + 0.45 * estimate.strength)
        span = self.config.velocity_ceiling - self.config.velocity_floor
        return round(self.config.velocity_floor + normalized * span)

    def _advance_state(
        self,
        estimates: dict[int, PitchEstimate],
        stream_time: float,
        latency_ms: float,
    ) -> list[FlashMidiEvent]:
        events: list[FlashMidiEvent] = []
        seen = set(estimates)

        # Notes currently present in the frame.
        for note, estimate in estimates.items():
            if note in self._active:
                active = self._active[note]
                active.confidence = estimate.confidence
                active.missing_frames = 0
                continue

            pending = self._pending.setdefault(note, _PendingNote())
            pending.frames += 1
            pending.confidence = max(pending.confidence, estimate.confidence)
            pending.strength = max(pending.strength, estimate.strength)
            if pending.frames < self.config.attack_frames:
                continue

            velocity = self._velocity(
                PitchEstimate(note, pending.confidence, pending.strength)
            )
            self._active[note] = _ActiveNote(
                velocity=velocity,
                confidence=pending.confidence,
            )
            del self._pending[note]
            self.stats.note_ons += 1
            events.append(
                FlashMidiEvent(
                    type="note_on",
                    note=note,
                    velocity=velocity,
                    stream_time=stream_time,
                    confidence=estimate.confidence,
                    latency_ms=latency_ms,
                )
            )

        # Drop attack candidates that did not persist into this frame.
        for note in list(self._pending):
            if note not in seen:
                del self._pending[note]

        # Active notes get a short release grace period to prevent FFT flicker
        # from producing rapid note-off/note-on chatter.
        for note in list(self._active):
            if note in seen:
                continue
            active = self._active[note]
            active.missing_frames += 1
            if active.missing_frames < self.config.release_frames:
                continue

            del self._active[note]
            self.stats.note_offs += 1
            events.append(
                FlashMidiEvent(
                    type="note_off",
                    note=note,
                    velocity=0,
                    stream_time=stream_time,
                    confidence=active.confidence,
                    latency_ms=latency_ms,
                )
            )
        return events

    def process_block(self, samples: np.ndarray) -> list[FlashMidiEvent]:
        """Consume mono float PCM and return any MIDI state changes."""
        block = np.asarray(samples, dtype=np.float32)
        if block.ndim == 2:
            block = np.mean(block, axis=1, dtype=np.float32)
        elif block.ndim != 1:
            raise ValueError("samples must be mono [frames] or [frames, channels]")
        if block.size == 0:
            return []

        self._input = np.concatenate((self._input, block))
        emitted: list[FlashMidiEvent] = []
        hop = self.config.hop_samples

        while self._input.size >= hop:
            step = self._input[:hop]
            self._input = self._input[hop:]

            if hop >= self._ring.size:
                self._ring[:] = step[-self._ring.size :]
            else:
                self._ring[:-hop] = self._ring[hop:]
                self._ring[-hop:] = step
            self._ring_filled = min(self._ring.size, self._ring_filled + hop)
            self._samples_consumed += hop

            if self._ring_filled < self._ring.size:
                continue

            started = time.perf_counter()
            estimates = self.detector.detect(self._ring)
            compute_ms = (time.perf_counter() - started) * 1000.0
            latency_ms = self.config.nominal_note_on_latency_ms + compute_ms

            self.stats.analyses += 1
            self.stats.last_compute_ms = compute_ms
            self.stats.max_compute_ms = max(self.stats.max_compute_ms, compute_ms)
            if latency_ms > self.config.latency_budget_ms:
                self.stats.budget_misses += 1

            stream_time = self._samples_consumed / self.config.sample_rate
            emitted.extend(self._advance_state(estimates, stream_time, latency_ms))

        return emitted

    def all_notes_off(self) -> list[FlashMidiEvent]:
        """Flush currently active notes, useful on shutdown/device errors."""
        stream_time = self._samples_consumed / self.config.sample_rate
        events = [
            FlashMidiEvent(
                type="note_off",
                note=note,
                velocity=0,
                stream_time=stream_time,
                confidence=active.confidence,
                latency_ms=0.0,
            )
            for note, active in sorted(self._active.items())
        ]
        self.stats.note_offs += len(events)
        self._active.clear()
        self._pending.clear()
        return events


class MidoMidiSink:
    """Send FlashMidiEvent objects to a system MIDI output port."""

    def __init__(self, port_name: str | None = None, virtual: bool = False):
        try:
            import mido
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Flash MIDI output requires mido (included with muscriptor)"
            ) from exc

        # python-rtmidi is the cross-platform backend used by the Flash extra.
        try:
            mido.set_backend("mido.backends.rtmidi")
            names = list(mido.get_output_names())
        except Exception as exc:  # pragma: no cover - host dependent
            raise RuntimeError(
                "No RtMidi backend is available. Install muscriptor[flash]."
            ) from exc

        if virtual:
            self._port = mido.open_output(port_name or "MuScripter Flash", virtual=True)
            self.name = port_name or "MuScripter Flash"
            return

        if port_name is None:
            if len(names) != 1:
                available = ", ".join(names) if names else "(none)"
                raise RuntimeError(
                    "Choose --midi-port because there is not exactly one MIDI "
                    f"output. Available: {available}"
                )
            port_name = names[0]
        elif port_name not in names:
            matches = [name for name in names if port_name.lower() in name.lower()]
            if len(matches) == 1:
                port_name = matches[0]
            else:
                available = ", ".join(names) if names else "(none)"
                raise RuntimeError(
                    f"MIDI output {port_name!r} was not found. Available: {available}"
                )

        self._port = mido.open_output(port_name)
        self.name = port_name

    def send(self, event: FlashMidiEvent) -> None:
        import mido

        if event.type == "note_on":
            message = mido.Message(
                "note_on", note=event.note, velocity=event.velocity, channel=0
            )
        else:
            message = mido.Message("note_off", note=event.note, velocity=0, channel=0)
        self._port.send(message)

    def close(self) -> None:
        self._port.close()


@dataclass
class LiveFlashSession:
    """Microphone -> FlashEngine -> MIDI bridge.

    ``sounddevice`` and ``python-rtmidi`` are imported lazily so the normal
    MuScriptor package/server does not acquire audio-device dependencies.
    """

    engine: FlashEngine
    sink: MidoMidiSink
    input_device: int | str | None = None
    _queue: queue.Queue[np.ndarray] = field(
        default_factory=lambda: queue.Queue(maxsize=32), init=False
    )
    dropped_blocks: int = field(default=0, init=False)

    def run(self, verbose: bool = False) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Live Flash input requires sounddevice. Install muscriptor[flash]."
            ) from exc

        hop = self.engine.config.hop_samples

        def callback(indata, frames, time_info, status):
            block = np.asarray(indata[:, 0], dtype=np.float32).copy()
            try:
                self._queue.put_nowait(block)
            except queue.Full:
                self.dropped_blocks += 1

        try:
            with sd.InputStream(
                samplerate=self.engine.config.sample_rate,
                blocksize=hop,
                channels=1,
                dtype="float32",
                device=self.input_device,
                callback=callback,
            ):
                while True:
                    block = self._queue.get()
                    for event in self.engine.process_block(block):
                        self.sink.send(event)
                        if verbose:
                            print(
                                f"{event.type:8s} {event.note:3d} "
                                f"vel={event.velocity:3d} "
                                f"conf={event.confidence:.2f} "
                                f"lat={event.latency_ms:.1f}ms"
                            )
        finally:
            for event in self.engine.all_notes_off():
                self.sink.send(event)
            self.sink.close()


def list_audio_devices() -> list[str]:
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Audio device listing requires sounddevice. Install muscriptor[flash]."
        ) from exc

    devices = sd.query_devices()
    result: list[str] = []
    for index, device in enumerate(devices):
        if int(device["max_input_channels"]) > 0:
            result.append(f"{index}: {device['name']}")
    return result


def list_midi_outputs() -> list[str]:
    try:
        import mido

        mido.set_backend("mido.backends.rtmidi")
        return list(mido.get_output_names())
    except Exception as exc:  # pragma: no cover - host dependent
        raise RuntimeError(
            "MIDI device listing requires python-rtmidi. Install muscriptor[flash]."
        ) from exc
