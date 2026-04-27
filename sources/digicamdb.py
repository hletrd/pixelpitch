"""
Digicamdb (digicamdb.com) — alias for openMVG.

The digicamdb live site is Cloudflare-protected (interactive JS challenge),
so direct scraping is not viable in CI. The same author (Gregor Brdnik)
seeded the openMVG/CameraSensorSizeDatabase with this data, and that
GitHub repository is MIT-licensed and continues to receive community
contributions. We use that as the digicamdb proxy.

If you want to live-scrape digicamdb, you would need DrissionPage with a
configured Cloudflare-solving session — see the existing browser path in
pixelpitch._create_browser. That capability is intentionally out of scope
for this module.
"""

from __future__ import annotations

from typing import Optional

from . import Spec
from .openmvg import fetch as _openmvg_fetch


def fetch(limit: Optional[int] = None) -> list[Spec]:
    specs = _openmvg_fetch(limit=limit)
    # Re-tag category so downstream consumers can distinguish provenance
    return [
        Spec(
            name=s.name,
            category=s.category,  # inherit openMVG's heuristic category
            type=s.type,
            size=s.size,
            pitch=s.pitch,
            mpix=s.mpix,
            year=s.year,
        )
        for s in specs
    ]


if __name__ == "__main__":
    rows = fetch(limit=5)
    for r in rows:
        print(r)
    print(f"... total: {len(fetch())}")
