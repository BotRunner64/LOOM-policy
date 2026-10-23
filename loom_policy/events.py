"""Public inference interface. Times are seconds, never frame indices."""
from dataclasses import dataclass
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
class Query:
    segment_id: str
    time: float
    offsets: tuple[float, ...]

    def __post_init__(self):
        if not math.isfinite(self.time) or not self.offsets:
            raise ValueError("Query needs a finite time and at least one offset")
        if any(not math.isfinite(x) or x <= 0 for x in self.offsets):
            raise ValueError("Offsets must be finite positive seconds")
        if any(a >= b for a, b in zip(self.offsets, self.offsets[1:])):
            raise ValueError("Offsets must be strictly increasing")


@dataclass(frozen=True)
class PolicyInput:
    segments: list[Segment]
    query: Query

    def visible(self) -> list[Segment]:
        ids = [s.segment_id for s in self.segments]
        sources = [s.source_id for s in self.segments]
        if not ids or any(not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("segment_id must be nonempty and unique")
        if any(not x for x in sources) or len(set(sources)) != len(sources):
            raise ValueError("Each physical source must occur once; never relabel target future as reference")
        if self.query.segment_id not in ids:
            raise ValueError("Query segment is absent")
        result = []
        for seg in self.segments:
            events = [e for e in seg.events if seg.segment_id != self.query.segment_id
                      or e.time <= self.query.time]
            # No temporal/modality embeddings for sequence order: use actual time,
            # explicit text index, and patch positions. Sort only for reproducibility.
            events.sort(key=lambda e: (e.time, e.stream_id, e.modality))
            result.append(Segment(seg.segment_id, seg.source_id, events))
        if not any(s.events for s in result):
            raise ValueError("At least one visible event is required")
        return result
