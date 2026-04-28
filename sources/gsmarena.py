"""
GSMArena (gsmarena.com) — smartphone main-camera sensors.

Phone spec pages list a "Main Camera" row containing for each lens:
  "<MP> MP, f/<aperture>, <focal>mm (wide|tele|...), <1/x.y>", <pitch>µm, ..."

We extract the first WIDE entry (i.e. the rear main camera). Sensor
fractional-inch format is converted to mm using the same TYPE_SIZE table
used elsewhere in pixelpitch.

Discovery: makers.php3 → per-brand listing pages (paginated) →
per-phone spec page.

robots.txt: User-agent: * is allowed for the paths we hit.
"""

from __future__ import annotations

import html as html_lib
import re
import time
from typing import Optional

from . import Spec, http_get, normalise_name, parse_year, TYPE_FRACTIONAL_RE

# Import the central sensor-size lookup table (now includes phone formats)
from pixelpitch import TYPE_SIZE as SENSOR_TYPE_SIZE

BASE = "https://www.gsmarena.com/"

# Phone listing
MAKER_RE = re.compile(r'href="([a-z0-9_]+-phones-(\d+)\.php)"')
PHONE_LINK_RE = re.compile(r'<a href="([a-z0-9_]+-\d+\.php)">')
PAGINATION_RE = re.compile(r'href="([a-z0-9_]+-phones-f-\d+-\d+\.php)"')

# Spec table
SPEC_ROW_RE = re.compile(
    r'<td[^>]*ttl[^>]*>(?:<a[^>]*>)?([^<]+?)(?:</a>)?</td>\s*'
    r'<td[^>]*nfo[^>]*>(.+?)</td>',
    re.DOTALL | re.IGNORECASE,
)

# Lens entry inside Main Camera value
LENS_RE = re.compile(
    r"(?P<mp>[\d.]+)\s*MP[^,]*,"  # 50 MP,
    r"\s*f/(?P<f>[\d.]+)[^,]*,"   # f/1.7,
    r"[^,]*?(?P<role>wide|ultrawide|ultra ?wide|telephoto|tele|periscope|macro|depth)?",
    re.IGNORECASE,
)
PITCH_RE = re.compile(r"([\d.]+)\s*(?:µm|μm|um)", re.IGNORECASE)

# fractional inch → (width_mm, height_mm). Uses the central TYPE_SIZE table
# from pixelpitch which now includes phone-only formats.
# A shallow copy prevents accidental mutation of the central TYPE_SIZE table.
PHONE_TYPE_SIZE: dict[str, tuple[float, float]] = dict(SENSOR_TYPE_SIZE)


def _parse_spec_table(html: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for m in SPEC_ROW_RE.finditer(html):
        label = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        value = re.sub(r"<[^>]+>", " ", m.group(2)).strip()
        value = html_lib.unescape(value)
        value = re.sub(r"\s+", " ", value)
        if label and value:
            fields[label] = value
    return fields


def _select_main_lens(camera_value: str) -> Optional[str]:
    """Pick the main (wide) lens entry from a GSMArena Main Camera value.

    Lenses are separated by newlines / spaces in our flattened text;
    the structure is roughly:
        "<lens1>\\n<lens2>\\n...\\n<features>"
    The main lens is the first one that has "(wide)" or no role tag,
    excluding ultrawide / tele / macro / depth roles.
    """
    if not camera_value:
        return None
    # GSMArena uses <br> between lens entries; we already collapsed to spaces
    # but the original "\n" survives in some cases. Try splitting on multiple
    # heuristics.
    raw = camera_value.replace("\n", " ")
    # Each lens entry tends to start with "<num> MP". Split on this token.
    parts = re.split(r"(?=\b\d+(?:\.\d+)?\s*MP\b)", raw)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return None

    def role_priority(p: str) -> int:
        pl = p.lower()
        if "(wide)" in pl:
            return 0
        if any(r in pl for r in ("(ultrawide)", "(ultra wide)", "ultra-wide")):
            return 3
        if any(r in pl for r in ("telephoto", "tele)", "periscope", "tele,")):
            return 4
        if any(r in pl for r in ("(macro)", "(depth)", "(tof)")):
            return 5
        # No explicit role tag — could be the main, give it priority 1
        return 1

    parts.sort(key=role_priority)
    return parts[0]


def _phone_to_spec(name: str, fields: dict[str, str]) -> Optional[Spec]:
    cam = (
        fields.get("Main Camera")
        or fields.get("Triple")
        or fields.get("Quad")
        or fields.get("Dual")
        or fields.get("Single")
        or ""
    )
    if not cam:
        # Some pages have the lens info under e.g. "Triple" without a
        # "Main Camera" row. Fall back to scanning all fields.
        for k, v in fields.items():
            if "MP" in v and re.search(r"\d+\s*MP.*?µm", v):
                cam = v
                break

    main = _select_main_lens(cam)
    if not main:
        return None

    mp_match = re.match(r"\s*([\d.]+)\s*MP", main)
    mpix = float(mp_match.group(1)) if mp_match else None

    pitch_match = PITCH_RE.search(main)
    pitch = float(pitch_match.group(1)) if pitch_match else None

    fmt_match = TYPE_FRACTIONAL_RE.search(main)
    sensor_type = fmt_match.group(1) if fmt_match else None
    size = PHONE_TYPE_SIZE.get(sensor_type) if sensor_type else None

    # Year from "Released" / "Announced" field
    year = None
    for k in ("Released", "Status", "Announced"):
        if fields.get(k):
            year = parse_year(fields[k])
            if year:
                break

    if not (mpix or pitch or size):
        return None

    return Spec(
        name=name,
        category="smartphone",
        type=sensor_type,
        size=size,
        pitch=pitch,
        mpix=mpix,
        year=year,
    )


def fetch_phone(slug: str) -> Optional[Spec]:
    """Fetch a single phone spec page by its slug (e.g. 'samsung_galaxy_s24_ultra-12771.php')."""
    url = BASE + slug
    body = http_get(url)
    if not body:
        return None
    fields = _parse_spec_table(body)
    name_m = re.search(r"<h1[^>]*>([^<]+)</h1>", body)
    name = normalise_name(html_lib.unescape(name_m.group(1))) if name_m else slug
    return _phone_to_spec(name, fields)


def _list_brand(brand_url: str, max_pages: int = 2) -> list[str]:
    """Enumerate phone slugs for a brand, paging up to max_pages."""
    slugs: list[str] = []
    body = http_get(BASE + brand_url)
    if not body:
        return slugs
    slugs.extend(PHONE_LINK_RE.findall(body))

    pages = list(set(PAGINATION_RE.findall(body)))[:max_pages]
    for p in pages:
        b = http_get(BASE + p)
        if b:
            slugs.extend(PHONE_LINK_RE.findall(b))
        time.sleep(0.5)

    seen: set[str] = set()
    out: list[str] = []
    for s in slugs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def fetch(
    limit: Optional[int] = None,
    sleep_seconds: float = 1.0,
    brands: Optional[list[str]] = None,
    max_pages_per_brand: int = 2,
) -> list[Spec]:
    """Crawl smartphones.

    Default brand list is the 12 brands most likely to publish full sensor
    specs. Increase max_pages_per_brand to dig deeper into the catalog.
    """
    if brands is None:
        brands = [
            "samsung-phones-9.php",
            "apple-phones-48.php",
            "google-phones-107.php",
            "xiaomi-phones-80.php",
            "oppo-phones-82.php",
            "vivo-phones-98.php",
            "oneplus-phones-95.php",
            "huawei-phones-58.php",
            "honor-phones-121.php",
            "sony-phones-7.php",
            "nothing-phones-128.php",
            "asus-phones-46.php",
        ]

    slugs: list[str] = []
    for b in brands:
        slugs.extend(_list_brand(b, max_pages=max_pages_per_brand))
        time.sleep(sleep_seconds)

    seen: set[str] = set()
    slugs = [s for s in slugs if not (s in seen or seen.add(s))]

    print(f"  gsmarena: {len(slugs)} phone slugs")
    if limit is not None:
        slugs = slugs[:limit]

    specs: list[Spec] = []
    for i, s in enumerate(slugs):
        spec = fetch_phone(s)
        if spec:
            specs.append(spec)
        if (i + 1) % 25 == 0:
            print(f"  gsmarena: {i + 1}/{len(slugs)} fetched, kept {len(specs)}")
        time.sleep(sleep_seconds)
    return specs


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith(".php"):
        s = fetch_phone(sys.argv[1])
        print(s)
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        rows = fetch(limit=n, sleep_seconds=0.5, max_pages_per_brand=1, brands=["samsung-phones-9.php"])
        for r in rows:
            print(r)
        print(f"... fetched {len(rows)} of {n}")
