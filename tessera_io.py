"""Minimal reader for the public TESSERA v1.1 global Icechunk store.

Reads only the chunks that are needed, so a small map window doesn't download
whole tiles. Anonymous access, no account needed.
"""

from __future__ import annotations

STORE_URL = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
_store = None


def store():
    """Shared IcechunkStore (opened lazily on first use)."""
    global _store
    if _store is None:
        from geotessera.icechunk import IcechunkStore

        _store = IcechunkStore(STORE_URL)
    return _store


def utm_zone(lon: float) -> int:
    return int((lon + 180) // 6) + 1


def zone_grid(zone: int):
    """Origin and pixel size of a UTM zone's grid: (x0, y0, res)."""
    ds = store().open_zone(zone)
    t = ds.attrs["spatial:transform"]  # [res, 0, x0, 0, -res, y0]
    return t[2], t[5], t[0]
