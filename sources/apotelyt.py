"""
Apotelyt (apotelyt.com).

Per-camera "sensor-pixels" pages list explicit fields:
  - Camera Model
  - Camera Type / Sensor Technology / Sensor Format
  - Sensor Size  : "35.9 x 23.9 mm"
  - Sensor Resolution : "32.7 Megapixels"
  - Image Resolution  : "7008 x 4672 pixels"
  - Pixel Pitch  : "5.12 µm"
  - Launch Date  : "October 2021"

URL discovery via apotelyt.com/sitemap-NN.xml/ index.
robots.txt allows all paths.
"""

from __future__ import annotations

import html as html_lib
import re
import time
from typing import Optional

from . import Spec, http_get, normalise_name, parse_year
from . import SIZE_MM_RE as SIZE_RE, PITCH_UM_RE as PITCH_RE, MPIX_RE

SITEMAP_INDEX = "https://apotelyt.com/sitemap-00.xml/"
LOC_RE = re.compile(r"<loc>([^<]+)</loc>")

# Each row collapses to: "Label | | | | Value"
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def _gather_urls() -> list[str]:
    body = http_get(SITEMAP_INDEX)
    if not body:
        return []
    sub_sitemaps = LOC_RE.findall(body)
    urls: list[str] = []
    for sm in sub_sitemaps:
        sb = http_get(sm)
        if not sb:
            continue
        for u in LOC_RE.findall(sb):
            if "sensor-pixels" in u:
                urls.append(u)
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _row_to_pair(row_html: str) -> Optional[tuple[str, str]]:
    text = TAG_RE.sub("|", row_html)
    text = html_lib.unescape(text)
    parts = [p.strip() for p in text.split("|") if p.strip()]
    if len(parts) < 2:
        return None
    return parts[0], " ".join(parts[1:])


def _body_category(camera_type: str, sensor_format: str, name: str) -> str:
    """Map Apotelyt's "Camera Type" + "Sensor Format" to a site category."""
    ct = (camera_type or "").lower()
    sf = (sensor_format or "").lower()

    if "mirrorless" in ct or "milc" in ct or "compact system" in ct:
        return "mirrorless"
    if "dslr" in ct or "slr" in ct or "reflex" in ct:
        return "dslr"
    if "rangefinder" in ct:
        return "rangefinder"
    if "action" in ct:
        return "actioncam"
    if "camcorder" in ct or "video" in ct:
        return "camcorder"
    if "compact" in ct or "bridge" in ct or "fixed" in ct or "point" in ct:
        return "fixed"

    if "full frame" in sf or "full-frame" in sf or "aps-c" in sf or "aps c" in sf or \
       "four thirds" in sf or "medium format" in sf:
        return "mirrorless"
    if "/" in sf and "1/" in sf:
        return "fixed"

    n = name.lower()
    if "gopro" in n or "insta360" in n or "osmo action" in n:
        return "actioncam"
    return "fixed"


def fetch_one(url: str) -> Optional[Spec]:
    body = http_get(url)
    if not body:
        return None

    fields: dict[str, str] = {}
    for m in ROW_RE.finditer(body):
        pair = _row_to_pair(m.group(1))
        if pair:
            label, value = pair
            fields.setdefault(label, value)

    name = fields.get("Camera Model") or ""
    name = normalise_name(name)
    if not name:
        return None

    size = None
    m = SIZE_RE.search(fields.get("Sensor Size", ""))
    if m:
        try:
            size = (float(m.group(1)), float(m.group(2)))
        except ValueError:
            size = None

    pitch = None
    m = PITCH_RE.search(fields.get("Pixel Pitch", ""))
    if m:
        try:
            pitch = float(m.group(1))
        except ValueError:
            pitch = None

    mpix = None
    m = MPIX_RE.search(fields.get("Sensor Resolution", ""))
    if m:
        try:
            mpix = float(m.group(1))
        except ValueError:
            mpix = None

    year = parse_year(fields.get("Launch Date", ""))

    sensor_fmt = fields.get("Sensor Format", "")
    camera_type = fields.get("Camera Type", "")
    category = _body_category(camera_type, sensor_fmt, name)

    if not (size or pitch or mpix):
        return None

    return Spec(
        name=name,
        category=category,
        type=None,
        size=size,
        pitch=pitch,
        mpix=mpix,
        year=year,
    )


def fetch(limit: Optional[int] = None, sleep_seconds: float = 0.4) -> list[Spec]:
    urls = _gather_urls()
    print(f"  apotelyt: {len(urls)} sensor-pixels URLs")
    if limit is not None:
        urls = urls[:limit]
    specs: list[Spec] = []
    for i, u in enumerate(urls):
        try:
            s = fetch_one(u)
            if s:
                specs.append(s)
        except Exception as ex:
            print(f"  apotelyt: failed {u}: {ex}")
        if (i + 1) % 25 == 0:
            print(f"  apotelyt: {i + 1}/{len(urls)} fetched, kept {len(specs)}")
        time.sleep(sleep_seconds)
    return specs


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rows = fetch(limit=n, sleep_seconds=0.2)
    for r in rows:
        print(r)
    print(f"... fetched {len(rows)} of {n}")
