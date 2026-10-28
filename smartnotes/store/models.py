from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Note:
    id: str            # "N001"... stable and never reused
    text: str
    category: str      # the four-label model's best guess, even when uncertain
    uncertain: bool
    confidence: float
    source: str        # "user" or "starter"
    created_at: str    # ISO 8601, UTC
    label: str = ""    # what the app shows (core label, Health/Family/Home/Work, or Other); "" = not set yet
    label_confidence: float = 0.0  # confidence behind `label`; 0 = not set yet

    @property
    def display_label(self) -> str:
        return self.label or self.category

    def to_dict(self) -> dict:
        return {**asdict(self), "label": self.display_label,
                "label_confidence": self.label_confidence or self.confidence}

    @property
    def number(self) -> int:
        return int(self.id[1:])
