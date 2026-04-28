"""
Imaging Resource (imaging-resource.com).

Per-camera spec pages explicitly list "Approximate Pixel Pitch: X microns"
alongside "Sensor size: A.AAmm2 (B.BBmm x C.CCmm)" and "Effective Megapixels".

Coverage: cameras the site has reviewed. URL discovery via sitemap_index.xml.

robots.txt: User-agent: * is allowed (only specific AI bots are blocked).
We use a generic UA out of courtesy.
"""

from __future__ import annotations

import html as html_lib
import re
import time
from typing import Optional

from . import (
    Spec,
    http_get,
    normalise_name,
    parse_year,
)

SITEMAP_INDEX = "https://www.imaging-resource.com/sitemap_index.xml"
LOC_RE = re.compile(r"<loc>([^<]+)</loc>")
REVIEW_URL_RE = re.compile(
    r"https://www\.imaging-resource\.com/cameras/[a-z0-9-]+-review/?$"
)
LEGACY_SPEC_URL_RE = re.compile(
    r"https://www\.imaging-resource\.com/cameras/[^/]+-specifications/?$"
)

LI_FIELD_RE = re.compile(
    r"<li>\s*([^:<]+?)\s*:\s*(.+?)(?=<li>|</ul>)", re.DOTALL | re.IGNORECASE
)

# "Sensor size: 366.6mm2 (23.50mm x 15.60mm)"
IR_SENSOR_SIZE_RE = re.compile(
    r"([\d.]+)\s*mm\s*[x×]\s*([\d.]+)\s*mm", re.IGNORECASE
)
# "Approximate Pixel Pitch: 3.92 microns"
IR_PITCH_RE = re.compile(r"([\d.]+)\s*microns?", re.IGNORECASE)
# "Effective Megapixels: 24.2"
IR_MPIX_RE = re.compile(r"(\d+\.?\d*)")
# "Date Available: 2021-08-31"
IR_DATE_RE = re.compile(r"(\d{4})[-/](\d{2})[-/](\d{2})")


def _gather_sitemaps() -> list[str]:
    body = http_get(SITEMAP_INDEX)
    if not body:
        return []
    return [u for u in LOC_RE.findall(body) if "review-cameras" in u]


def _gather_review_urls() -> list[str]:
    """Return list of {model}-review/ URLs (modern) and *-specifications/ (legacy).

    The sitemap is oldest-first; we reverse so callers passing --limit N
    sample the most recent cameras first.
    """
    urls: list[str] = []
    for sm in _gather_sitemaps():
        body = http_get(sm)
        if not body:
            continue
        for u in LOC_RE.findall(body):
            if REVIEW_URL_RE.fullmatch(u) or LEGACY_SPEC_URL_RE.fullmatch(u):
                urls.append(u)
    seen: set[str] = set()
    out: list[str] = []
    for u in reversed(urls):
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _spec_url(review_url: str) -> str:
    if "-specifications" in review_url:
        return review_url
    if not review_url.endswith("/"):
        review_url += "/"
    return review_url + "specifications/"


def _parse_fields(html: str) -> dict[str, str]:
    """Extract <li>label: value<li> fields from the spec body."""
    fields: dict[str, str] = {}
    for m in LI_FIELD_RE.finditer(html):
        label = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        value = re.sub(r"<[^>]+>", " ", m.group(2)).strip().rstrip("</")
        value = html_lib.unescape(value).strip()
        if label and value:
            fields.setdefault(label, value)
    return fields


def _body_category(camera_format: str, sensor_format: str, name: str) -> str:
    """Map IR's "Camera Format" / "Sensor Format" fields to one of the
    pixelpitch site categories.
    """
    cf = (camera_format or "").lower()
    sf = (sensor_format or "").lower()

    if "compact system camera" in cf or "mirrorless" in cf:
        return "mirrorless"
    if "digital slr" in cf or "dslr" in cf or "slr" in cf:
        return "dslr"
    if "rangefinder" in cf:
        return "rangefinder"
    if "action" in cf:
        return "actioncam"
    if "camcorder" in cf or "video" in cf:
        return "camcorder"
    if "compact" in cf or "fixed" in cf or "bridge" in cf or "point" in cf:
        return "fixed"

    # Fallbacks by sensor format hints
    if "/" in sf and "1/" in sf:
        return "fixed"
    if "full frame" in sf or "full-frame" in sf or sf == "35mm" or "aps-c" in sf or "aps c" in sf or \
       "micro four" in sf or "four thirds" in sf or "medium format" in sf:
        # No camera-type hint; modern interchangeable-lens systems are
        # overwhelmingly mirrorless in 2025+, so guess mirrorless.
        return "mirrorless"

    n = name.lower()
    if "gopro" in n or "insta360" in n or "osmo action" in n or "actioncam" in n:
        return "actioncam"
    if "camcorder" in n or "handycam" in n:
        return "camcorder"

    return "fixed"


def _parse_camera_name(fields: dict[str, str], fallback_url: str) -> Optional[str]:
    """Extract a clean camera name.

    Sony spec pages use SKU-style "Model Name" values like
    "Sony Alpha ILCE-A7 IV" or "Sony Alpha ILCZV-E10". We normalise these
    to "Sony A7 IV" / "Sony ZV-E10" using the URL slug, which contains
    the marketing name directly.
    """
    name = (fields.get("Model Name") or "").strip()

    if name.lower().startswith("sony"):
        parts = fallback_url.rstrip("/").rsplit("/")
        slug = parts[-1]
        if slug == "specifications":
            # Modern format: .../camera-slug-review/specifications/
            slug = parts[-2]
        # Strip known URL suffixes (works for both modern and legacy formats)
        slug = re.sub(r"-(review|specifications|digital-camera-review-information.*)$", "", slug)
        cleaned = slug.replace("-", " ").title()
        cleaned = re.sub(
            r"\b(Ii|Iii|Iv|Vi|Vii|Viii|Ix)\b",
            lambda m: m.group(1).upper(),
            cleaned,
        )
        cleaned = cleaned.replace("Sony Zv ", "Sony ZV-")
        # Sony uppercase series: .title() mangles multi-letter prefixes
        # (fx->Fx, rx->Rx, etc.).  Restore the correct ALL-CAPS form.
        cleaned = re.sub(r"\bFx(\d)", r"FX\1", cleaned)
        cleaned = re.sub(r"\bRx(\d)", r"RX\1", cleaned)
        cleaned = re.sub(r"\bHx(\d)", r"HX\1", cleaned)
        cleaned = re.sub(r"\bWx(\d)", r"WX\1", cleaned)
        cleaned = re.sub(r"\bTx(\d)", r"TX\1", cleaned)
        cleaned = re.sub(r"\bQx(\d)", r"QX\1", cleaned)
        cleaned = re.sub(r"\bDsc\b", r"DSC", cleaned)
        # Normalise DSC-hyphen to DSC-space so names are consistent
        # whether derived from Model Name ("DSC-HX400") or URL slug
        # ("dsc-hx400" → "Dsc Hx400" after .replace("-"," ") + .title()).
        cleaned = re.sub(r"\bDSC-", "DSC ", cleaned)
        return normalise_name(cleaned)

    if name:
        return normalise_name(name)

    parts = fallback_url.rstrip("/").rsplit("/")
    slug = parts[-1]
    if slug == "specifications":
        slug = parts[-2]
    slug = re.sub(r"-(review|specifications|digital-camera-review-information.*)$", "", slug)
    cleaned = slug.replace("-", " ").title()
    # Apply Sony-specific normalizations when the URL contains "sony-"
    if "sony-" in fallback_url.lower():
        cleaned = re.sub(
            r"\b(Ii|Iii|Iv|Vi|Vii|Viii|Ix)\b",
            lambda m: m.group(1).upper(),
            cleaned,
        )
        cleaned = cleaned.replace("Sony Zv ", "Sony ZV-")
        # Sony uppercase series: .title() mangles multi-letter prefixes
        cleaned = re.sub(r"\bFx(\d)", r"FX\1", cleaned)
        cleaned = re.sub(r"\bRx(\d)", r"RX\1", cleaned)
        cleaned = re.sub(r"\bHx(\d)", r"HX\1", cleaned)
        cleaned = re.sub(r"\bWx(\d)", r"WX\1", cleaned)
        cleaned = re.sub(r"\bTx(\d)", r"TX\1", cleaned)
        cleaned = re.sub(r"\bQx(\d)", r"QX\1", cleaned)
        cleaned = re.sub(r"\bDsc\b", r"DSC", cleaned)
        # Normalise DSC-hyphen to DSC-space for consistency
        cleaned = re.sub(r"\bDSC-", "DSC ", cleaned)
    return normalise_name(cleaned)


def fetch_one(spec_url: str) -> Optional[Spec]:
    body = http_get(spec_url)
    if not body:
        return None

    fields = _parse_fields(body)
    if not fields:
        return None

    name = _parse_camera_name(fields, spec_url)
    if not name:
        return None

    # Sensor size
    size = None
    sensor_size_text = fields.get("Sensor size") or fields.get("Sensor Size") or ""
    m = IR_SENSOR_SIZE_RE.search(sensor_size_text)
    if m:
        try:
            size = (float(m.group(1)), float(m.group(2)))
        except ValueError:
            size = None

    # Pixel pitch
    pitch = None
    pitch_text = fields.get("Approximate Pixel Pitch") or ""
    m = IR_PITCH_RE.search(pitch_text)
    if m:
        pitch = float(m.group(1))

    # Megapixels
    mpix = None
    mp_text = fields.get("Effective Megapixels") or fields.get("Megapixels") or ""
    m = IR_MPIX_RE.search(mp_text)
    if m:
        try:
            mpix = float(m.group(1))
        except ValueError:
            mpix = None

    # Year
    year = None
    date_text = fields.get("Date Available") or fields.get("Date available") or ""
    year = parse_year(date_text)

    sensor_fmt = fields.get("Sensor Format") or fields.get("Sensor format") or ""
    camera_fmt = fields.get("Camera Format") or fields.get("Camera format") or ""
    category = _body_category(camera_fmt, sensor_fmt, name)

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


def fetch(limit: Optional[int] = None, sleep_seconds: float = 0.5) -> list[Spec]:
    urls = _gather_review_urls()
    print(f"  imaging-resource: {len(urls)} review URLs")
    if limit is not None:
        urls = urls[:limit]

    specs: list[Spec] = []
    for i, u in enumerate(urls):
        spec = fetch_one(_spec_url(u))
        if spec:
            specs.append(spec)
        if (i + 1) % 25 == 0:
            print(f"  imaging-resource: {i + 1}/{len(urls)} fetched, kept {len(specs)}")
        time.sleep(sleep_seconds)
    return specs


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rows = fetch(limit=n, sleep_seconds=0.2)
    for r in rows:
        print(r)
    print(f"... fetched {len(rows)} of {n}")
