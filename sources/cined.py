"""
CineD Camera Database (cined.com/camera-database/).

Cinema/hybrid cameras with sensor dimensions, recording modes and lab-tested
dynamic range. The site renders the database client-side via JavaScript
(no static HTML rows, no public WP REST endpoint), so we drive a real
browser via DrissionPage — the same path Geizhals already uses in
pixelpitch._create_browser.

Coverage per camera page:
  - Sensor Size  (e.g. "Super 35", "Full Frame", "Micro Four Thirds",
                  sometimes with explicit mm in the General Data section)
  - Sensor mode resolutions (mode -> pixel WxH)
  - Lens Mount, base ISO, etc.

Pixel pitch is derived (sensor_width_mm / sensor_pixels_w) when both
fields are present. If only the format class is given (no explicit mm
dimensions), we leave ``spec.size`` as None so that ``merge_camera_data``
can preserve more accurate measured values from Geizhals when available.
The template will show "unknown" for sensor size when no measured data
exists for the camera — this is more honest than presenting FORMAT_TO_MM
approximations as if they were measured. The ``FORMAT_TO_MM`` table is
kept for the regex coverage test only.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from . import Spec, normalise_name, parse_year, SIZE_MM_RE as SIZE_RE

DATABASE_URL = "https://www.cined.com/camera-database/"
RES_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")

FORMAT_TO_MM: dict[str, tuple[float, float]] = {
    "full frame": (36.0, 24.0),
    "super 35": (24.89, 18.66),
    "super 35 mm": (24.89, 18.66),
    "super35": (24.89, 18.66),
    "aps-c": (23.6, 15.7),
    "micro four thirds": (17.3, 13.0),
    "four thirds": (17.3, 13.0),
    "1\"": (13.2, 8.8),
    "1-inch": (13.2, 8.8),
    "1 inch": (13.2, 8.8),
    "2/3\"": (8.8, 6.6),
    "2/3-inch": (8.8, 6.6),
    "medium format": (43.8, 32.9),
}


def _create_browser():
    """Reuse the existing DrissionPage browser helper from pixelpitch.

    Lazy-imported so this module remains importable even when DrissionPage
    is not installed (e.g. in test/CI environments that only need other
    sources). Will raise ImportError when actually called.
    """
    from pixelpitch import _create_browser as build  # type: ignore
    return build()


def _collect_camera_links(page, max_scrolls: int = 30) -> list[str]:
    """Scroll the camera-database listing to load all entries, then collect URLs."""
    page.get(DATABASE_URL)
    time.sleep(5)
    seen: set[str] = set()
    for _ in range(max_scrolls):
        page.scroll.to_bottom()
        time.sleep(1.5)
        for a in page.eles("tag:a"):
            href = a.attr("href") or ""
            if "/camera-database/" in href and href.rstrip("/") != DATABASE_URL.rstrip("/"):
                seen.add(href.split("?")[0].split("#")[0])
        if len(seen) >= 200:
            break
    return sorted(seen)


def _parse_camera_page(page, url: str) -> Optional[Spec]:
    page.get(url)
    time.sleep(3)
    text = page.html
    body_text = re.sub(r"<[^>]+>", " ", text)
    body_text = re.sub(r"\s+", " ", body_text)

    name_m = re.search(r"<h1[^>]*>([^<]+)</h1>", text)
    name = normalise_name(name_m.group(1)) if name_m else url.rstrip("/").rsplit("/", 1)[-1]

    fmt_m = re.search(
        r"(Full Frame|Super[- ]?35(?:\s*mm)?|APS-C|Micro Four Thirds|Four Thirds|1\"|1[- ]inch|2/3\"|2/3[- ]inch|Medium Format)",
        body_text,
        re.IGNORECASE,
    )
    fmt = fmt_m.group(1) if fmt_m else ""

    size = None
    s = SIZE_RE.search(body_text)
    if s:
        try:
            size = (float(s.group(1)), float(s.group(2)))
        except ValueError:
            size = None
    if size is None and fmt:
        # Don't set spec.size from FORMAT_TO_MM lookup — the lookup
        # provides approximate dimensions from the format class name.
        # Setting spec.size from the lookup prevents merge_camera_data
        # from preserving more accurate measured values from Geizhals
        # (because the merge only preserves existing spec.size when
        # new spec.size is None). Leave spec.size = None; the template
        # will show "unknown" for sensor size when no Geizhals data
        # exists, which is more honest than showing an approximation
        # as if it were measured data.
        # Note: we also don't set spec.type because format class names
        # like "Super 35" or "APS-C" are not fractional-inch types that
        # TYPE_SIZE understands.
        pass

    pixels = None
    px = RES_RE.search(body_text)
    if px:
        pw, ph = int(px.group(1)), int(px.group(2))
        if 1000 <= pw <= 20000 and 500 <= ph <= 15000:
            pixels = (pw, ph)

    mpix = round(pixels[0] * pixels[1] / 1_000_000, 1) if pixels else None
    year_m = re.search(r"Release Date.{0,40}?(\d{4})", body_text, re.IGNORECASE)
    if year_m:
        y = int(year_m.group(1))
        year = y if 1900 <= y <= 2100 else None
    else:
        year = parse_year(body_text[:500])

    if not (size or mpix):
        return None

    return Spec(
        name=name,
        category="cinema",
        type=None,
        size=size,
        pitch=None,
        mpix=mpix,
        year=year,
    )


def fetch(limit: Optional[int] = None) -> list[Spec]:
    """Fetch cinema camera specs via DrissionPage. Requires browser dependency."""
    try:
        page = _create_browser()
    except ImportError as e:
        print(f"  cined: skipping — DrissionPage not installed ({e})")
        return []
    except Exception as e:
        print(f"  cined: skipping — browser init failed ({e})")
        return []

    try:
        urls = _collect_camera_links(page)
        print(f"  cined: {len(urls)} camera URLs")
        if limit is not None:
            urls = urls[:limit]
        specs: list[Spec] = []
        for i, u in enumerate(urls):
            try:
                s = _parse_camera_page(page, u)
                if s:
                    specs.append(s)
            except Exception as ex:
                print(f"  cined: failed {u}: {ex}")
            if (i + 1) % 10 == 0:
                print(f"  cined: {i + 1}/{len(urls)} fetched, kept {len(specs)}")
        return specs
    finally:
        try:
            page.quit()
        except Exception:
            pass


if __name__ == "__main__":
    rows = fetch(limit=5)
    for r in rows:
        print(r)
