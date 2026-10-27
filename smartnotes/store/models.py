from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Note:
    id: str            # "N001"... stable and never reused
    text: str
    category: str      # the model's best guess, even when uncertain
    uncertain: bool
    confidence: float
    source: str        # "user" or "starter"
    created_at: str    # ISO 8601, UTC

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def number(self) -> int:
        return int(self.id[1:])
