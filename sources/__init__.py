"""
Alternative data sources for camera sensor / pixel-pitch information.

Each module exposes:

    fetch(limit: Optional[int] = None) -> list[Spec]

returning records compatible with pixelpitch.Spec so they can flow through
derive_spec() and write_csv() unchanged.

Note: some source modules accept additional keyword arguments beyond
``limit`` (e.g. ``sleep_seconds``, ``brands``, ``max_pages_per_brand``).
These are documented in the individual module docstrings.

Sources implemented:
  - openmvg          : MIT-licensed CSV (primary, bulk)
  - digicamdb        : upstream of openmvg; live site is Cloudflare-blocked,
                       so this module is a thin alias / documentation
  - imaging_resource : per-camera spec pages (most reliable for current models)
  - apotelyt         : per-camera "sensor-pixels" pages (good gap-filler)
  - gsmarena         : smartphone main-camera sensor specs
  - cined            : cinema-camera database (JS-rendered, requires browser)
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import Spec  # noqa: E402

# Polite, generic UA. Imaging Resource blocks AI-named UAs in robots.txt
# (anthropic-ai, Claude-Web, GPTBot...) but allows User-agent: *.
USER_AGENT = (
    "Mozilla/5.0 (compatible; pixelpitch-fetcher/1.0; "
    "+https://hletrd.github.io/pixelpitch/)"
)


def http_get(url: str, timeout: float = 30.0, retries: int = 3) -> Optional[str]:
    """Plain HTTP GET with retry. Returns body text or None."""
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read().decode(charset, errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    print(f"  GET failed: {url} ({last_err})", file=sys.stderr)
    return None


# Helper regex patterns used by several sources — single source of truth.
# SIZE_MM_RE matches "Ax Bmm" with ASCII x, Unicode ×, and optional spaces.
# PITCH_UM_RE matches pixel pitch values with these suffixes:
#   µm (micro sign), um (ASCII), microns/micron, μm (Greek mu),
#   &micro;m and &#956;m (HTML entities).
SIZE_MM_RE = re.compile(r"([\d.]+)\s*[x×]\s*([\d.]+)\s*mm", re.IGNORECASE)
PITCH_UM_RE = re.compile(r"([\d.]+)\s*(?:µm|um|microns?|μm|&micro;m|&#0?956;m)", re.IGNORECASE)
MPIX_RE = re.compile(r"([\d.]+)\s*(?:effective\s+)?(?:Mega ?pixels?|MP)", re.IGNORECASE)
TYPE_FRACTIONAL_RE = re.compile(r"(1/[\d.]+)(?:\"|\s*inch|-inch|-type|\s*type|″)", re.IGNORECASE)


def parse_year(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return int(m.group(1)) if m else None


def normalise_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    return name


__all__ = [
    "Spec",
    "USER_AGENT",
    "http_get",
    "SIZE_MM_RE",
    "PITCH_UM_RE",
    "MPIX_RE",
    "TYPE_FRACTIONAL_RE",
    "parse_year",
    "normalise_name",
]
