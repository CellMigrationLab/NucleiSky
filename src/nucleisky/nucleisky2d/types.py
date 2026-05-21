
"""types.py Lightweight shared types."""

from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class PixelSizeDetails:
    source: Optional[str] = None
    x_um: Optional[float] = None
    y_um: Optional[float] = None
    z_um: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BBox:
    """
    Canonical bounding box in pixel coordinates using half-open intervals.

        y in [y0, y1)
        x in [x0, x1)
    """
    y0: int
    y1: int
    x0: int
    x1: int

    def __post_init__(self):
        object.__setattr__(self, "y0", int(self.y0))
        object.__setattr__(self, "y1", int(self.y1))
        object.__setattr__(self, "x0", int(self.x0))
        object.__setattr__(self, "x1", int(self.x1))
        if self.y1 < self.y0 or self.x1 < self.x0:
            raise ValueError(f"Invalid BBox: {self}")

    # --- add these two methods ---
    def __iter__(self):
        # Allows: y0, y1, x0, x1 = bbox  (and map(int, bbox))
        yield int(self.y0)
        yield int(self.y1)
        yield int(self.x0)
        yield int(self.x1)

    def __len__(self):
        return 4

    @property
    def height(self) -> int:
        return int(self.y1 - self.y0)

    @property
    def width(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def empty(self) -> bool:
        return self.height <= 0 or self.width <= 0

    def as_y0y1x0x1(self) -> tuple[int, int, int, int]:
        return (int(self.y0), int(self.y1), int(self.x0), int(self.x1))

    def as_y0x0y1x1(self) -> tuple[int, int, int, int]:
        return (int(self.y0), int(self.x0), int(self.y1), int(self.x1))

    def slices_yx(self) -> tuple[slice, slice]:
        return (slice(self.y0, self.y1), slice(self.x0, self.x1))

    def pad(self, margin: int) -> "BBox":
        m = int(margin)
        return BBox(self.y0 - m, self.y1 + m, self.x0 - m, self.x1 + m)

    def clamp(self, shape_yx: tuple[int, int], *, min_size: int = 1) -> "BBox":
        H, W = map(int, shape_yx)
        ms = max(1, int(min_size))

        y0 = max(0, min(H, self.y0))
        y1 = max(0, min(H, self.y1))
        x0 = max(0, min(W, self.x0))
        x1 = max(0, min(W, self.x1))

        if (y1 - y0) < ms:
            y1 = min(H, y0 + ms)
            y0 = max(0, y1 - ms)
        if (x1 - x0) < ms:
            x1 = min(W, x0 + ms)
            x0 = max(0, x1 - ms)

        return BBox(y0, y1, x0, x1)

    @staticmethod
    def from_y0x0y1x1(t: tuple[int, int, int, int]) -> "BBox":
        y0, x0, y1, x1 = map(int, t)
        return BBox(y0, y1, x0, x1)

