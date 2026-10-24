"""Public inference interface. Times are seconds, never frame indices."""
from dataclasses import dataclass, field
from typing import Any, Literal
import math


@dataclass(frozen=True)
class Event:
    time: float
    stream_id: str
    modality: Literal["image", "state", "text"]
    value: Any  # image CHW float [0,1], state [26], or nonempty UTF-8 text

    def __post_init__(self):
        if not math.isfinite(self.time) or not self.stream_id:
            raise ValueError("An event needs a finite time and nonempty stream_id")
        if self.modality not in ("image", "state", "text"):
            raise ValueError(f"Unknown modality: {self.modality}")


@dataclass(frozen=True)
class Segment:
    segment_id: str
    source_id: str  # physical trajectory identity, e.g. dataset path + episode_id
    events: list[Event]


@dataclass(frozen=True)
class ChunkConfig:
    """Fixed output timing, shared by the model and label construction."""
    size: int = 8
    interval_ns: int = 50_000_000

    def __post_init__(self):
        if type(self.size) is not int or self.size < 1 or type(self.interval_ns) is not int or self.interval_ns < 1:
            raise ValueError("Chunk size and interval_ns must be positive integers")
        if self.size * self.interval_ns > 2**63 - 1:
            raise ValueError("Chunk duration exceeds int64 nanoseconds")

    @property
    def offsets_ns(self):
        return tuple(self.interval_ns * (h + 1) for h in range(self.size))


@dataclass(frozen=True)
class PolicyInput:
    current: Segment
    time: float
    references: list[Segment] = field(default_factory=list)

    def __post_init__(self):
        if not math.isfinite(self.time):
            raise ValueError("Current time must be finite seconds")

    def visible(self) -> list[Segment]:
        segments = [*self.references, self.current]
        ids = [s.segment_id for s in segments]
        sources = [s.source_id for s in segments]
        if not ids or any(not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("segment_id must be nonempty and unique")
        if any(not x for x in sources) or len(set(sources)) != len(sources):
            raise ValueError("Each physical source must occur once; current trajectory cannot also be a reference")
        result = []
        for seg in segments:
            events = [e for e in seg.events if seg.segment_id != self.current.segment_id
                      or e.time <= self.time]
            # No temporal/modality embeddings for sequence order: use actual time,
            # explicit text index, and patch positions. Sort only for reproducibility.
            events.sort(key=lambda e: (e.time, e.stream_id, e.modality))
            result.append(Segment(seg.segment_id, seg.source_id, events))
        if not result[-1].events:
            raise ValueError("Current trajectory needs at least one visible observation")
        return result
