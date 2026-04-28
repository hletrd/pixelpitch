"""
Camera sensor pixel pitch database — calculates and aggregates pixel pitch
from multiple sources including geizhals.eu, Imaging Resource, Apotelyt,
GSMArena, CineD, the openMVG camera sensor database, and Digicamdb (via
openMVG).
"""

import csv
import html
import io
import json
import os
import re
import sys

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Optional, Tuple, List
from urllib.parse import quote_plus

from models import Spec, SpecDerived

SCRIPT_DIR = Path(__file__).resolve().parent

# For fixed-lens cameras we assume 4:3 sensor aspect ratio if not given.
# Also, the following mapping of given sensor sizes to sensor areas is used from wikipedia:
# http://en.wikipedia.org/wiki/Image_sensor_format
# This seems necessary as the advertised sensor sizes are often larger than they actually are.

FIXED_URL = "https://geizhals.eu/?cat=dcam&hloc=at&hloc=de&hloc=pl&hloc=uk&hloc=eu&fcols=1418&fcols=86&fcols=3377&sort=artikel"

# For DSLR and Mirrorless cameras we use the specified sensor dimensions as is.
DSLR_URL = "https://geizhals.eu/?cat=dcamsp&xf=1480_Spiegelreflex+(DSLR)&hloc=de&hloc=pl&hloc=uk&hloc=eu&fcols=166&fcols=5761&fcols=3378&sort=artikel"
MIRRORLESS_URL = "https://geizhals.eu/?cat=dcamsp&xf=1480_Spiegellos+(DSLM)&hloc=de&hloc=pl&hloc=uk&hloc=eu&fcols=166&fcols=5761&fcols=3378&sort=artikel"
RANGEFINDER_URL = "https://geizhals.eu/?cat=dcamsp&xf=1480_Messsucher&hloc=de&hloc=pl&hloc=uk&hloc=eu&fcols=166&fcols=5761&fcols=3378&sort=artikel"
CAMCORDER_URL = "https://geizhals.eu/?cat=dvcam&hloc=de&hloc=pl&hloc=uk&hloc=eu&fcols=205&fcols=195&fcols=3373&sort=artikel"
ACTIONCAM_URL = "https://geizhals.eu/?cat=dvcamac&hloc=de&hloc=pl&hloc=uk&hloc=eu&fcols=5023&fcols=5025&fcols=5036&sort=artikel"

SIZE_RE = re.compile(r"([\d\.]+)x([\d\.]+)mm")
SENSOR_TYPE_RE = re.compile(r"(1/[\d\.]+)\"")
PITCH_RE = re.compile(r"([\d\.]+)µm")
MPIX_RE = re.compile(r"([\d\.]+)\s*Megapixel")

# from http://en.wikipedia.org/wiki/Image_sensor_format
TYPE_SIZE: dict[str, Tuple[float, float]] = {
    # Compact / phone sensor formats (fractional-inch optical format)
    "1/4.0": (3.60, 2.70),
    "1/3.6": (4.00, 3.00),
    "1/3.2": (4.54, 3.42),
    "1/3": (4.80, 3.60),
    "1/2.93": (4.91, 3.68),
    "1/2.8": (5.12, 3.84),
    "1/2.76": (5.20, 3.90),
    "1/2.7": (5.37, 4.04),
    "1/2.55": (5.65, 4.24),
    "1/2.5": (5.76, 4.29),
    "1/2.3": (6.17, 4.55),
    "1/2": (6.40, 4.80),
    "1/1.95": (6.56, 4.92),
    "1/1.9": (6.72, 5.04),
    "1/1.8": (7.18, 5.32),
    "1/1.78": (7.20, 5.40),
    "1/1.74": (7.36, 5.52),
    "1/1.7": (7.60, 5.70),
    "1/1.65": (7.76, 5.82),
    "1/1.6": (8.08, 6.01),
    "1/1.56": (8.20, 6.16),
    "1/1.5": (8.80, 6.60),
    "1/1.43": (8.96, 6.72),
    "1/1.4": (9.14, 6.86),
    "1/1.35": (9.50, 7.10),
    "1/1.33": (9.62, 7.22),
    "1/1.31": (9.79, 7.34),
    "1/1.3": (9.84, 7.40),
    "1/1.29": (9.92, 7.45),
    "1/1.28": (10.00, 7.50),
    "1/1.22": (10.50, 7.87),
    "1/1.2": (10.67, 8.00),
    "1/1.12": (11.43, 8.57),
    "1/1.1": (11.65, 8.74),
    "1/1.07": (11.96, 8.97),
    "1/1.0": (12.80, 9.60),
    "1": (13.20, 8.80),
}

EXTRAS = [
    "weiß",
    "schwarz",
    "rot",
    "grau",
    "pink",
    "gold",
    "silber",
    "violett",
    "grün",
    "blau",
    "orange",
    "braun",
    "gelb",
    "beige",
    "bordeaux",
    "bronze",
    "rosa",
    "graphit",
    "titan",
    "camouflage",
    "khaki",
    "anthrazit",
    "mit Objektiv",
    "Gehäuse",
    "Body",
]
EXTRAS_RE = re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, EXTRAS)))
PARENS_RE = re.compile(r"\([^()]+\)$")

# Regex for parsing the new Svelte SPA HTML
ROW_RE = re.compile(
    r'<tr class="datatable__row[^"]*"[^>]*>(.*?)</tr>', re.DOTALL
)
DD_TITLE_RE = re.compile(r'<dd[^>]*title="([^"]*)"')
DT_TITLE_RE = re.compile(r'<dt[^>]*title="([^"]*)"')
DATA_NAME_RE = re.compile(r'data-name="([^"]+)"')


def sensor_area(width: float, height: float) -> float:
    return width * height


def sensor_size(diag: float, aspect: float) -> Tuple[float, float]:
    diagmm = diag * 25.4
    h = sqrt(diagmm**2 / (aspect**2 + 1))
    w = aspect * h
    return w, h


def sensor_size_from_type(
    typ: Optional[str],
) -> Optional[Tuple[float, float]]:
    """Convert a fractional-inch sensor type designation to (width_mm, height_mm).

    The lookup table contains measured (actual) sensor dimensions which are
    significantly more accurate than computing from the nominal diagonal.
    When the type is in the table, the table value is always used.

    For types not in the table (e.g. "1/3.1"), the sensor size is computed
    from the nominal diagonal.  Note that computed values are approximations
    because the optical format naming convention does not represent the actual
    sensor diagonal.

    Invalid fractional types (e.g. "1/0", "1/") return None instead of
    raising ZeroDivisionError or ValueError.
    """
    if not typ:
        return None

    # Always prefer the lookup table — it has measured dimensions.
    if typ in TYPE_SIZE:
        return TYPE_SIZE[typ]

    if typ.startswith("1/"):
        try:
            diag = 1 / float(typ[2:])
        except (ZeroDivisionError, ValueError):
            return None
        if diag <= 0:
            return None
        size = sensor_size(diag, 4 / 3)
        return size

    return None


def pixel_pitch(area: float, mpix: float) -> float:
    return 1000 * sqrt(area / (mpix * 10**6))


def load_sensors_database() -> dict:
    """Load the sensor database from sensors.json.

    Expected schema::

        {
            "IMX455": {
                "sensor_width_mm": 36.0,
                "sensor_height_mm": 24.0,
                "megapixels": [61.2, 61.0]
            },
            ...
        }

    Each key is a sensor model name.  ``megapixels`` is a list because
    the same physical sensor may be used at different resolutions.
    """
    try:
        with open(SCRIPT_DIR / "sensors.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not load sensors.json: {e}")
        return {}


def match_sensors(
    width: Optional[float],
    height: Optional[float],
    megapixels: Optional[float],
    sensors_db: dict,
    size_tolerance: float = 2,
    megapixel_tolerance: float = 5,
) -> List[str]:
    if not sensors_db or not width or not height:
        return []

    matches = []

    for sensor_name, sensor_data in sensors_db.items():
        sensor_width = sensor_data.get("sensor_width_mm")
        sensor_height = sensor_data.get("sensor_height_mm")
        sensor_megapixels = sensor_data.get("megapixels", [])

        if not sensor_width or not sensor_height:
            continue

        width_match = abs(width - sensor_width) / width * 100 <= size_tolerance
        height_match = abs(height - sensor_height) / height * 100 <= size_tolerance

        if not (width_match and height_match):
            continue

        if megapixels is not None and sensor_megapixels:
            megapixel_match = any(
                abs(megapixels - mp) / megapixels * 100 <= megapixel_tolerance
                for mp in sensor_megapixels
            )
            if megapixel_match:
                matches.append(sensor_name)
        elif megapixels is None or not sensor_megapixels:
            # No megapixel data available — match on size alone (lower confidence)
            matches.append(sensor_name)

    return sorted(matches)


def load_csv(output_dir: Path) -> Optional[str]:
    path = output_dir / "camera-data.csv"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"Warning: Could not read {path}: {e}")
            return None

    return None


def parse_existing_csv(csv_content: str) -> List[SpecDerived]:
    """Parse a CSV string produced by write_csv back into SpecDerived objects.

    Uses the csv module for correct RFC 4180 handling (quoted fields, escaped
    quotes, commas inside values).  The header row determines whether an ``id``
    column is present; all subsequent rows are mapped by column index relative
    to the detected schema.
    """
    if not csv_content:
        return []

    # Strip UTF-8 BOM if present. Excel's "CSV UTF-8" save option adds a
    # BOM (﻿) at the start of the file, which would make header[0] = "﻿id"
    # instead of "id", breaking schema detection and causing 0-row parses.
    if csv_content[0] == "﻿":
        csv_content = csv_content[1:]

    specs: List[SpecDerived] = []
    reader = csv.reader(io.StringIO(csv_content))
    rows = list(reader)

    if len(rows) < 2:
        return []

    header = rows[0]
    has_id = header[0] == "id" if header else False

    for values in rows[1:]:
        if not values or not any(v.strip() for v in values):
            continue

        try:
            # Normalise column count — pad short rows with empty strings
            if has_id:
                while len(values) < 10:
                    values.append("")
                record_id = int(values[0]) if values[0] else None
                name = values[1].strip()
                category = values[2].strip()
                type_str = values[3].strip() if values[3].strip() else None
                width_str = values[4]
                height_str = values[5]
                area_str = values[6]
                mpix_str = values[7]
                pitch_str = values[8]
                year_str = values[9]
                sensors_str = values[10] if len(values) > 10 else ""
            else:
                while len(values) < 9:
                    values.append("")
                record_id = None
                name = values[0].strip()
                category = values[1].strip()
                type_str = values[2].strip() if values[2].strip() else None
                width_str = values[3]
                height_str = values[4]
                area_str = values[5]
                mpix_str = values[6]
                pitch_str = values[7]
                year_str = values[8]
                sensors_str = values[9] if len(values) > 9 else ""

            size = None
            if width_str and height_str:
                try:
                    width = float(width_str)
                    height = float(height_str)
                    size = (width, height)
                except ValueError:
                    pass

            area = float(area_str) if area_str else None
            mpix = float(mpix_str) if mpix_str else None
            pitch = float(pitch_str) if pitch_str else None
            year = int(year_str) if year_str else None
            matched_sensors = sensors_str.split(";") if sensors_str else []

            spec = Spec(name, category, type_str, size, pitch, mpix, year)
            derived = SpecDerived(spec, size, area, pitch, matched_sensors, record_id)
            specs.append(derived)

        except Exception as e:
            line_preview = ",".join(values)[:50]
            print(f"Error parsing CSV line: {line_preview}... - {e}")
            continue

    return specs


def create_camera_key(spec: Spec) -> str:
    """Create a deduplication key from name + category.

    Year is deliberately excluded so that cameras with missing or
    differing years across sources are correctly merged into a single
    entry.  Year preservation is handled by merge_camera_data.
    """
    return f"{spec.name.lower().strip()}-{spec.category}"


def merge_camera_data(
    new_specs: List[SpecDerived], existing_specs: List[SpecDerived]
) -> List[SpecDerived]:
    """Merge new camera data with existing data.

    Cameras are matched by ``name + category`` (via ``create_camera_key``).
    If the same key appears multiple times in ``new_specs`` (e.g., the same
    camera from both Geizhals and a source CSV), only the first occurrence
    is kept and subsequent duplicates are silently dropped.  This prevents
    duplicate rows on the All Cameras page.
    """
    print(
        f"Merging {len(new_specs)} new records with {len(existing_specs)} existing records"
    )

    existing_by_key = {}
    for spec in existing_specs:
        key = create_camera_key(spec.spec)
        existing_by_key[key] = spec

    found_keys = set()
    merged_specs = []
    seen_new_keys = set()

    for new_spec in new_specs:
        key = create_camera_key(new_spec.spec)

        # Deduplicate among new_specs: if we've already processed this key
        # from new data, skip the duplicate.
        if key in seen_new_keys:
            continue
        seen_new_keys.add(key)

        found_keys.add(key)

        if key in existing_by_key:
            existing_spec = existing_by_key[key]
            new_spec.id = existing_spec.id
            # Preserve year from existing data if new data has none
            if new_spec.spec.year is None and existing_spec.spec.year is not None:
                new_spec.spec.year = existing_spec.spec.year
            elif (
                new_spec.spec.year is not None
                and existing_spec.spec.year is not None
                and new_spec.spec.year != existing_spec.spec.year
            ):
                print(
                    f"  Year changed for {new_spec.spec.name[:40]}: "
                    f"{existing_spec.spec.year} -> {new_spec.spec.year}"
                )
            print(f"Updated existing camera: {new_spec.spec.name[:50]}")

        merged_specs.append(new_spec)

    sensors_db: Optional[dict] = None  # lazy-loaded on first use

    for key, existing_spec in existing_by_key.items():
        if key not in found_keys:
            # Lazy-load sensors_db only when we have existing-only cameras
            # that need sensor matching. This avoids reading sensors.json
            # when all existing cameras are also in the new data.
            if existing_spec.size and sensors_db is None:
                sensors_db = load_sensors_database()
            if existing_spec.size and sensors_db:
                existing_spec.matched_sensors = match_sensors(
                    existing_spec.size[0],
                    existing_spec.size[1],
                    existing_spec.spec.mpix,
                    sensors_db,
                )
            print(f"Preserving removed camera: {existing_spec.spec.name[:50]}")
            merged_specs.append(existing_spec)

    merged_specs.sort(key=lambda x: x.spec.name.lower())

    for i, spec in enumerate(merged_specs):
        spec.id = i

    print(f"Final merged data contains {len(merged_specs)} records")
    return merged_specs


def _create_browser():
    """Create and return a DrissionPage browser instance."""
    from DrissionPage import ChromiumPage, ChromiumOptions  # lazy import
    co = ChromiumOptions()
    co.set_argument("--disable-blink-features=AutomationControlled")
    co.set_argument("--no-sandbox")

    # macOS: use Google Chrome (non-headless) to bypass Cloudflare
    mac_chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(mac_chrome):
        co.set_browser_path(mac_chrome)
        co.headless(False)
        co.set_argument("--remote-debugging-port=9222")
        co.set_argument("--remote-debugging-address=127.0.0.1")
    else:
        # Linux CI: use system Chrome
        chrome_path = "/opt/google/chrome/chrome"
        if os.path.exists(chrome_path):
            co.set_browser_path(chrome_path)

    page = ChromiumPage(co)
    page.set.load_mode.eager  # Don't wait for full page load
    # Navigate to homepage to solve Cloudflare challenge and set cookie
    page.get("https://geizhals.eu/")
    import time
    time.sleep(5)
    page.set.cookies("blaettern=1000")
    return page


def extract_entries(page, url: str) -> list[str]:
    """Fetch product rows from a URL."""
    import time

    print(f"Fetching {url}", flush=True)
    page.get(url)

    # Wait for product rows to appear (Cloudflare challenge may delay)
    rows = []
    for attempt in range(24):
        time.sleep(5)
        content = page.html
        rows = ROW_RE.findall(content)
        if rows:
            break
        # Debug: show what we got instead
        title_match = re.search(r"<title>(.*?)</title>", content)
        title = title_match.group(1) if title_match else "no title"
        print(f"  Waiting (attempt {attempt + 1})... page title: {title}", flush=True)

    if not rows:
        raise RuntimeError(f"No entries found at {url}")
    print(f"  Found {len(rows)} entries", flush=True)
    return rows


def parse_sensor_field(sensor_text: str) -> dict:
    """Parse the sensor description field from the new site.

    Examples:
        "Kleinbild, CMOS 36.0x24.0mm, 6.94µm Pixelgröße"
        "CMOS 1/3.1\", 1.09µm Pixelgröße"
        "APS-C, CMOS 23.5x15.6mm"
        "CMOS"
    """
    result = {"type": None, "size": None, "pitch": None}

    if not sensor_text:
        return result

    # Extract sensor type (e.g. "1/3.1", "1/2.3")
    type_match = SENSOR_TYPE_RE.search(sensor_text)
    if type_match:
        result["type"] = type_match.group(1)

    # Extract sensor dimensions (e.g. "36.0x24.0mm")
    size_match = SIZE_RE.search(sensor_text)
    if size_match:
        result["size"] = (float(size_match.group(1)), float(size_match.group(2)))

    # Extract pixel pitch (e.g. "6.94µm")
    pitch_match = PITCH_RE.search(sensor_text)
    if pitch_match:
        result["pitch"] = float(pitch_match.group(1))

    return result


def extract_specs(entries: list[str], category: str) -> list[Spec]:
    """Extract camera specifications from HTML table rows."""
    specs = []

    for row in entries:
        # Extract name
        name_match = DATA_NAME_RE.search(row)
        if not name_match:
            continue

        name = html.unescape(name_match.group(1))
        name = " ".join(name.split())

        # Extract dd/dt title pairs
        dd_titles = DD_TITLE_RE.findall(row)
        dt_titles = DT_TITLE_RE.findall(row)

        # Build a dict of field_name -> value
        fields = {}
        for dt_title, dd_title in zip(dt_titles, dd_titles):
            fields[dt_title] = dd_title

        # Parse megapixel
        mpix = None
        mpix_text = fields.get("Megapixel effektiv", "")
        if mpix_text:
            mpix_match = MPIX_RE.search(mpix_text)
            if mpix_match:
                mpix = float(mpix_match.group(1))

        # Parse sensor info
        sensor_text = fields.get("Sensor", "")
        sensor_info = parse_sensor_field(sensor_text)

        typ = sensor_info["type"]
        size = sensor_info["size"]
        pitch = sensor_info["pitch"]

        # Parse type from "Typ" field for interchangeable-lens cameras
        if typ is None:
            type_text = fields.get("Typ", "")
            if type_text and "/" in type_text:
                type_match = SENSOR_TYPE_RE.search(type_text)
                if type_match:
                    typ = type_match.group(1)

        specs.append(Spec(name, category, typ, size, pitch, mpix, year=None))

    specs = deduplicate_specs(specs)
    return specs


def deduplicate_specs(specs: list[Spec]) -> list[Spec]:
    """Unify product names and remove duplicates."""
    groups: dict[str, list[Spec]] = defaultdict(list)
    rest = []

    for spec in specs:
        match = EXTRAS_RE.search(spec.name)
        if match:
            unified_name = spec.name[: match.start()]
            groups[unified_name].append(spec)
        else:
            rest.append(spec)

    for unified_name, grouped_specs in groups.items():
        ref = grouped_specs[0]
        if all(
            spec.type == ref.type
            and spec.size == ref.size
            and spec.pitch == ref.pitch
            and spec.mpix == ref.mpix
            for spec in grouped_specs
        ):
            years = [s.year for s in grouped_specs if s.year]
            year = min(years) if years else None
            rest.append(
                Spec(
                    unified_name,
                    ref.category,
                    ref.type,
                    ref.size,
                    ref.pitch,
                    ref.mpix,
                    year,
                )
            )
        else:
            rest.extend(grouped_specs)

    def remove_parens(spec: Spec) -> Spec:
        name = spec.name.strip()
        match = PARENS_RE.search(name)
        if match:
            name = name[: match.start()].strip()
        return Spec(
            name, spec.category, spec.type, spec.size, spec.pitch, spec.mpix, spec.year
        )

    rest = list(map(remove_parens, rest))

    # Final deduplication: remove exact duplicates by (name, category, type, size, pitch, mpix)
    seen: set[tuple] = set()
    deduped: list[Spec] = []
    for spec in rest:
        key = (spec.name, spec.category, spec.type, spec.size, spec.pitch, spec.mpix)
        if key not in seen:
            seen.add(key)
            deduped.append(spec)

    return deduped


def derive_spec(
    spec: Spec, sensors_db: Optional[dict] = None
) -> SpecDerived:
    if spec.size is None:
        size = sensor_size_from_type(spec.type)
    else:
        size = spec.size

    if size is not None and spec.mpix is not None:
        area = size[0] * size[1]
    else:
        area = None

    if spec.pitch:
        pitch = spec.pitch
    elif spec.mpix is not None and area is not None:
        pitch = pixel_pitch(area, spec.mpix)
    else:
        pitch = None

    matched_sensors = []
    if sensors_db and size:
        matched_sensors = match_sensors(size[0], size[1], spec.mpix, sensors_db)

    return SpecDerived(spec, size, area, pitch, matched_sensors)


def derive_specs(specs: list[Spec]) -> list[SpecDerived]:
    sensors_db = load_sensors_database()
    return [derive_spec(spec, sensors_db) for spec in specs]


def get_category(page, url: str, category: str) -> list[SpecDerived]:
    """Fetch and derive specs for a camera category."""
    entries = extract_entries(page, url)
    return derive_specs(extract_specs(entries, category))


def sorted_by(
    specs: list[SpecDerived], key: str = "pitch", reverse: bool = True
) -> list[SpecDerived]:
    key_functions = {
        "pitch": lambda c: c.pitch if c.pitch else -1,
        "area": lambda c: c.area if c.area else -1,
        "mpix": lambda c: c.spec.mpix if c.spec.mpix else -1,
        "name": lambda c: c.spec.name,
    }
    return sorted(specs, key=key_functions[key], reverse=reverse)


def prettyprint(derived: SpecDerived) -> None:
    spec = derived.spec

    print(f'"{spec.name}": ', end="")

    if derived.size:
        print(f"{derived.size[0]:.1f}x{derived.size[1]:.1f}mm sensor", end="")
        if spec.size is None:
            print(f" (derived from type: {spec.type})", end="")
    else:
        print("unknown sensor size", end="")

    if spec.mpix:
        print(f", {spec.mpix:.1f} MP", end="")
    else:
        print(", unknown resolution", end="")

    if derived.pitch:
        print(f", {derived.pitch:.1f}µm pixel pitch", end="")

    print()


def datetimeformat(value, format="%d %b %Y %H:%M:%S UTC"):
    return value.strftime(format)


_env = None


def _get_env():
    """Lazy-init Jinja env so this module can be imported without jinja2."""
    global _env
    if _env is None:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        _env = Environment(
            loader=FileSystemLoader(SCRIPT_DIR / "templates"),
            autoescape=select_autoescape(["html", "xml"]),
        )
        _env.filters["formatdate"] = datetimeformat
        _env.filters["urlencode"] = quote_plus
    return _env


def write_csv(specs: list[SpecDerived], output_file: Path) -> None:
    """Write camera specs to a CSV file using the csv module for proper escaping."""
    print(f"Writing CSV to {output_file}")

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "name", "category", "type", "sensor_width_mm",
            "sensor_height_mm", "sensor_area_mm2", "megapixels",
            "pixel_pitch_um", "year", "matched_sensors",
        ])

        for derived in specs:
            spec = derived.spec

            id_str = str(derived.id) if derived.id is not None else ""
            category_str = spec.category or ""
            type_str = spec.type or ""
            width_str = f"{derived.size[0]:.2f}" if derived.size else ""
            height_str = f"{derived.size[1]:.2f}" if derived.size else ""
            area_str = f"{derived.area:.2f}" if derived.area else ""
            mpix_str = f"{spec.mpix:.1f}" if spec.mpix else ""
            pitch_str = f"{derived.pitch:.2f}" if derived.pitch else ""
            year_str = str(spec.year) if spec.year else ""
            sensors_str = (
                ";".join(derived.matched_sensors) if derived.matched_sensors else ""
            )

            writer.writerow([
                id_str, spec.name, category_str, type_str, width_str,
                height_str, area_str, mpix_str, pitch_str, year_str,
                sensors_str,
            ])


CATEGORIES = [
    (FIXED_URL, "fixed"),
    (DSLR_URL, "dslr"),
    (MIRRORLESS_URL, "mirrorless"),
    (RANGEFINDER_URL, "rangefinder"),
    (CAMCORDER_URL, "camcorder"),
    (ACTIONCAM_URL, "actioncam"),
]


def _load_per_source_csvs(output_dir: Path) -> List[SpecDerived]:
    """Read dist/camera-data-{source}.csv for every registered source.

    These are produced by `python pixelpitch.py source <name>` runs and
    serve as caches between deployments. Missing files are silently
    skipped — failure of one source must not block the build.
    """
    extras: List[SpecDerived] = []
    for src in SOURCE_REGISTRY:
        path = output_dir / f"camera-data-{src}.csv"
        if not path.exists():
            print(f"  source CSV missing: {path.name} (skipped)")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"  could not read {path.name}: {e}")
            continue
        parsed = parse_existing_csv(content)
        # Per-source CSVs carry their own ids — clear them so merge gives
        # globally unique ids.
        for d in parsed:
            d.id = None
        extras.extend(parsed)
        print(f"  loaded {len(parsed)} records from {path.name}")
    return extras


def render_html(output_dir: Path, skip_geizhals: bool = False) -> None:
    """Render all HTML files."""
    print("Loading previous CSV artifact...")
    previous_csv = load_csv(output_dir)
    existing_specs = parse_existing_csv(previous_csv) if previous_csv else []

    category_specs = {}
    if skip_geizhals:
        print("Skipping Geizhals fetch (per --skip-geizhals)")
        for _url, category in CATEGORIES:
            category_specs[category] = []
    else:
        print("Creating browser session...")
        page = _create_browser()

        print("Fetching camera data from Geizhals...")
        for url, category in CATEGORIES:
            try:
                category_specs[category] = get_category(page, url, category)
            except Exception as e:
                print(f"  Geizhals {category} failed: {e} — keeping previous data")
                category_specs[category] = []

        page.quit()

    # Geizhals "rangefinder" (Messsucher) filter misclassifies many non-rangefinder
    # cameras (Canon EOS DSLRs, Fujifilm mirrorless, etc.) that already appear under
    # the correct category. Remove rangefinder entries whose name also exists in any
    # other Geizhals category to prevent duplicate entries on the All Cameras page.
    rf_names = {spec.spec.name for spec in category_specs.get("rangefinder", [])}
    other_names = set()
    for cat, specs in category_specs.items():
        if cat != "rangefinder":
            other_names.update(spec.spec.name for spec in specs)
    dup_rf_names = rf_names & other_names
    if dup_rf_names:
        print(f"  Removing {len(dup_rf_names)} rangefinder duplicates (also in dslr/mirrorless/etc.)")
        category_specs["rangefinder"] = [
            s for s in category_specs.get("rangefinder", [])
            if s.spec.name not in dup_rf_names
        ]

    print("Loading per-source CSVs...")
    extra_specs = _load_per_source_csvs(output_dir)

    new_specs_all: List[SpecDerived] = []
    for specs in category_specs.values():
        new_specs_all.extend(specs)
    new_specs_all.extend(extra_specs)

    specs_all = merge_camera_data(new_specs_all, existing_specs)

    specs_by_category: dict[str, list[SpecDerived]] = {}
    # Geizhals categories: include only freshly fetched names so the page
    # reflects the live retailer data; older preserved cameras still appear
    # on the "all" page.
    for category in category_specs:
        new_names = {spec.spec.name for spec in category_specs[category]}
        specs_by_category[category] = sorted_by(
            [s for s in specs_all if s.spec.name in new_names],
            "pitch",
        )
    # New / source-driven categories: include every record with that
    # category from the merged set.
    for extra_category in ("smartphone", "cinema"):
        specs_by_category[extra_category] = sorted_by(
            [s for s in specs_all if s.spec.category == extra_category],
            "pitch",
        )
    # Backfill the Geizhals categories with source-only records (e.g. modern
    # mirrorless cameras that Geizhals dropped from its current listing).
    geizhals_names = {spec.spec.name for cat in category_specs for spec in category_specs[cat]}
    for cat in ("dslr", "mirrorless", "rangefinder", "fixed", "camcorder", "actioncam"):
        existing_names = {s.spec.name for s in specs_by_category[cat]}
        for s in specs_all:
            if (
                s.spec.category == cat
                and s.spec.name not in existing_names
                and s.spec.name not in geizhals_names
            ):
                specs_by_category[cat].append(s)
        specs_by_category[cat] = sorted_by(specs_by_category[cat], "pitch")
    specs_all = sorted_by(specs_all, "pitch")

    date = datetime.now(timezone.utc)

    print("Generating HTML files...")

    output_dir.mkdir(exist_ok=True)

    template = _get_env().get_template("pixelpitch.html")

    (output_dir / "fixedlens.html").write_text(
        template.render(
            title="Fixed-lens Cameras",
            specs=specs_by_category["fixed"],
            page="fixedlens",
            date=date,
        ),
        encoding="utf-8",
    )

    (output_dir / "dslr.html").write_text(
        template.render(
            title="DSLR Cameras",
            specs=specs_by_category["dslr"],
            page="dslr",
            date=date,
        ),
        encoding="utf-8",
    )

    (output_dir / "mirrorless.html").write_text(
        template.render(
            title="Mirrorless Cameras",
            specs=specs_by_category["mirrorless"],
            page="mirrorless",
            date=date,
        ),
        encoding="utf-8",
    )

    (output_dir / "rangefinder.html").write_text(
        template.render(
            title="Rangefinder Cameras",
            specs=specs_by_category["rangefinder"],
            page="rangefinder",
            date=date,
        ),
        encoding="utf-8",
    )

    (output_dir / "camcorder.html").write_text(
        template.render(
            title="Camcorders",
            specs=specs_by_category["camcorder"],
            page="camcorder",
            date=date,
        ),
        encoding="utf-8",
    )

    (output_dir / "actioncam.html").write_text(
        template.render(
            title="Actioncams",
            specs=specs_by_category["actioncam"],
            page="actioncam",
            date=date,
        ),
        encoding="utf-8",
    )

    (output_dir / "smartphone.html").write_text(
        template.render(
            title="Smartphone Cameras",
            specs=specs_by_category["smartphone"],
            page="smartphone",
            date=date,
        ),
        encoding="utf-8",
    )

    (output_dir / "cinema.html").write_text(
        template.render(
            title="Cinema Cameras",
            specs=specs_by_category["cinema"],
            page="cinema",
            date=date,
        ),
        encoding="utf-8",
    )

    (output_dir / "index.html").write_text(
        template.render(title="All Cameras", specs=specs_all, page="all", date=date),
        encoding="utf-8",
    )

    (output_dir / "about.html").write_text(
        _get_env().get_template("about.html").render(
            title="About Pixel Pitch", page="about", date=date
        ),
        encoding="utf-8",
    )

    write_csv(specs_all, output_dir / "camera-data.csv")

    static_seo_files = ["robots.txt"]
    for filename in static_seo_files:
        src_path = SCRIPT_DIR / filename
        if src_path.exists():
            (output_dir / filename).write_text(
                src_path.read_text(encoding="utf-8"), encoding="utf-8"
            )

    sitemap_path = SCRIPT_DIR / "sitemap.xml"
    if sitemap_path.exists():
        sitemap_content = sitemap_path.read_text(encoding="utf-8")
        sitemap_content = sitemap_content.replace("__LASTMOD__", date.strftime("%Y-%m-%d"))
        (output_dir / "sitemap.xml").write_text(sitemap_content, encoding="utf-8")

    print(f"HTML files written to {output_dir}")
    print(f'CSV file written to {output_dir / "camera-data.csv"}')
    print(f"SEO files copied to {output_dir}")


SOURCE_REGISTRY = {
    "openmvg": "sources.openmvg",
    "imaging-resource": "sources.imaging_resource",
    "apotelyt": "sources.apotelyt",
    "gsmarena": "sources.gsmarena",
    "cined": "sources.cined",
}


def fetch_source(name: str, limit: Optional[int], output_dir: Path) -> None:
    """Fetch records from an alternative source and write a CSV file."""
    import importlib

    if name not in SOURCE_REGISTRY:
        raise ValueError(
            f"Unknown source: {name}. "
            f"Available: {', '.join(sorted(SOURCE_REGISTRY))}"
        )

    module = importlib.import_module(SOURCE_REGISTRY[name])
    print(f"Fetching from source '{name}' (limit={limit})...")
    raw_specs = module.fetch(limit=limit) if limit is not None else module.fetch()

    derived = derive_specs(raw_specs)
    derived = sorted_by(derived, "pitch")
    for i, d in enumerate(derived):
        d.id = i

    output_dir.mkdir(exist_ok=True, parents=True)
    out_file = output_dir / f"camera-data-{name}.csv"
    write_csv(derived, out_file)
    print(f"  fetched {len(raw_specs)} records, wrote {out_file}")


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "html":
            args = sys.argv[2:]
            output_dir = Path("dist")
            skip_geizhals = False
            i = 0
            while i < len(args):
                a = args[i]
                if a == "--skip-geizhals":
                    skip_geizhals = True
                elif not a.startswith("--"):
                    output_dir = Path(a)
                i += 1
            render_html(output_dir, skip_geizhals=skip_geizhals)
        elif cmd == "source":
            if len(sys.argv) < 3:
                print(f"Usage: python pixelpitch.py source <name> [--limit N] [--out DIR]")
                print(f"Available sources: {', '.join(sorted(SOURCE_REGISTRY))}")
                sys.exit(1)
            src = sys.argv[2]
            limit: Optional[int] = None
            out_dir = Path("dist")
            args = sys.argv[3:]
            for i, a in enumerate(args):
                if a == "--limit" and i + 1 < len(args):
                    limit = int(args[i + 1])
                elif a == "--out" and i + 1 < len(args):
                    out_dir = Path(args[i + 1])
            fetch_source(src, limit, out_dir)
        elif cmd == "list":
            print("Fetching all cameras...")
            page = _create_browser()
            all_specs = []
            for url, category in CATEGORIES:
                all_specs.extend(get_category(page, url, category))
            page.quit()
            specs_sorted = sorted_by(all_specs, "pitch")
            for spec in specs_sorted:
                if spec.pitch:
                    prettyprint(spec)
        elif cmd in ("--help", "-h"):
            print("Usage: python pixelpitch.py [command] [args]")
            print("\nCommands:")
            print(
                "  html [dir]    Generate HTML files "
                "(default: current directory)"
            )
            print("  list          List all cameras with pixel pitch to console")
            print(
                "  source <name> [--limit N] [--out DIR]\n"
                "                Fetch from an alternative source.\n"
                f"                Available: {', '.join(sorted(SOURCE_REGISTRY))}"
            )
            print("  --help, -h    Show this help message")
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Run 'python pixelpitch.py --help' for usage information")
            sys.exit(1)
    else:
        render_html(Path("dist"))


if __name__ == "__main__":
    main()
