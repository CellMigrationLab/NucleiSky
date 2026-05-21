"""types.py Lightweight shared types for 3D workflows."""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class VoxelSizeDetails:
    """Provenance and metadata tracking for 3D physical sizes."""
    source: Optional[str] = None
    x_um: Optional[float] = None
    y_um: Optional[float] = None
    z_um: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BBox3D:
    """Canonical 3D bounding box in pixel coordinates using half-open intervals.
    
        z in [z0, z1)
        y in [y0, y1)
        x in [x0, x1)
    """

    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    def __post_init__(self):
        object.__setattr__(self, "z0", int(self.z0))
        object.__setattr__(self, "z1", int(self.z1))
        object.__setattr__(self, "y0", int(self.y0))
        object.__setattr__(self, "y1", int(self.y1))
        object.__setattr__(self, "x0", int(self.x0))
        object.__setattr__(self, "x1", int(self.x1))
        if self.z1 < self.z0 or self.y1 < self.y0 or self.x1 < self.x0:
            raise ValueError(f"Invalid BBox3D: {self}")

    def __iter__(self):
        yield int(self.z0)
        yield int(self.z1)
        yield int(self.y0)
        yield int(self.y1)
        yield int(self.x0)
        yield int(self.x1)

    def __len__(self):
        return 6

    @property
    def depth(self) -> int:
        return int(self.z1 - self.z0)

    @property
    def height(self) -> int:
        return int(self.y1 - self.y0)

    @property
    def width(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def empty(self) -> bool:
        return self.depth <= 0 or self.height <= 0 or self.width <= 0

    def slices_zyx(self) -> tuple[slice, slice, slice]:
        """Returns (slice(z0,z1), slice(y0,y1), slice(x0,x1)) for ZYX arrays."""
        return (slice(self.z0, self.z1), slice(self.y0, self.y1), slice(self.x0, self.x1))

    def pad(self, margin: int) -> "BBox3D":
        """Expands the bounding box uniformly by `margin` voxels."""
        m = int(margin)
        return BBox3D(
            self.z0 - m, self.z1 + m,
            self.y0 - m, self.y1 + m,
            self.x0 - m, self.x1 + m
        )

    def clamp(self, shape_zyx: tuple[int, int, int], min_size: int = 1) -> "BBox3D":
        """Restricts the bounding box to the given volume shape."""
        Z, Y, X = map(int, shape_zyx)
        ms = max(1, int(min_size))

        z0 = max(0, min(Z, self.z0))
        z1 = max(0, min(Z, self.z1))
        y0 = max(0, min(Y, self.y0))
        y1 = max(0, min(Y, self.y1))
        x0 = max(0, min(X, self.x0))
        x1 = max(0, min(X, self.x1))

        if (z1 - z0) < ms:
            z1 = min(Z, z0 + ms)
            z0 = max(0, z1 - ms)
        if (y1 - y0) < ms:
            y1 = min(Y, y0 + ms)
            y0 = max(0, y1 - ms)
        if (x1 - x0) < ms:
            x1 = min(X, x0 + ms)
            x0 = max(0, x1 - ms)

        return BBox3D(z0, z1, y0, y1, x0, x1)
