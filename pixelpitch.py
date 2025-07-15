"""
This script calculates pixel pitch in µm for cameras listed on geizhals.at.
"""

import html
import json
import os
import re
import sys
import tempfile
import zipfile

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, UTC
from math import sqrt
from pathlib import Path
from typing import Optional, Tuple, List
from urllib.parse import quote_plus

import requests

from jinja2 import Environment, FileSystemLoader, select_autoescape

# For fixed-lens cameras we assume 4:3 sensor aspect ratio if not given.
# Also, the following mapping of given sensor sizes to sensor areas is used from wikipedia:
# http://en.wikipedia.org/wiki/Image_sensor_format
# This seems necessary as the advertised sensor sizes are often larger than they actually are.

FIXED_URL = "https://geizhals.eu/?cat=dcam&hloc=at&hloc=de&hloc=pl&hloc=uk&hloc=eu&fcols=1418&fcols=86&fcols=3377&sort=artikel&bl1_id=1000"

# For DSLR and Mirrorless cameras we use the specified sensor dimensions as is.
DSLR_URL = "https://geizhals.eu/?cat=dcamsp&xf=1480_Spiegelreflex+(DSLR)&hloc=de&hloc=pl&hloc=uk&hloc=eu&fcols=166&fcols=5761&fcols=3378&sort=artikel&bl1_id=1000"
MIRRORLESS_URL = "https://geizhals.eu/?cat=dcamsp&xf=1480_Spiegellos+(DSLM)&hloc=de&hloc=pl&hloc=uk&hloc=eu&fcols=166&fcols=5761&fcols=3378&sort=artikel&bl1_id=1000"
RANGEFINDER_URL = "https://geizhals.eu/?cat=dcamsp&xf=1480_Messsucher&hloc=de&hloc=pl&hloc=uk&hloc=eu&fcols=166&fcols=5761&fcols=3378&sort=artikel&bl1_id=1000"

SIZE_RE = re.compile(r"\s([\d\.]+)x([\d\.]+)mm")
TYPE_RE = re.compile(
    r'<div class="productlist__additionalfilter">\s+(1/[\d\.]+)&quot;\s+</div>'
)
MPIX_RE = re.compile(
    r'<div class="productlist__additionalfilter">\s+([\d\.]+) Megapixel\s+</div>'
)
PITCH_RE = re.compile(
    r'<div class="productlist__additionalfilter">\s+([\d\.]+)&micro;m\s+</div>'
)
YEAR_RE = re.compile(
    r'<div class="productlist__additionalfilter">\s+([\d]{4})\s+</div>'
)
NAME_RE = re.compile(r'data-name="(.+?)"')

# from http://en.wikipedia.org/wiki/Image_sensor_format
TYPE_SIZE: dict[str, Tuple[float, float]] = {
    "1/3.2": (4.54, 3.42),
    "1/3": (4.80, 3.60),
    "1/2.7": (5.37, 4.04),
    "1/2.5": (5.76, 4.29),
    "1/2.3": (6.17, 4.55),
    "1/2": (6.40, 4.80),
    "1/1.8": (7.18, 5.32),
    "1/1.7": (7.60, 5.70),
    "1/1.6": (8.08, 6.01),
    "1/1.5": (8.80, 6.60),
    "1/1.2": (10.67, 8.00),
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
]
EXTRAS_RE = re.compile("|".join(EXTRAS))
PARENS_RE = re.compile(r"\(.+\)$")


@dataclass
class Spec:
    name: str
    type: Optional[str]
    size: Optional[Tuple[float, float]]
    pitch: Optional[float]
    mpix: Optional[float]
    year: Optional[int]


@dataclass
class SpecDerived:
    spec: Spec
    size: Optional[Tuple[float, float]]
    area: Optional[float]
    pitch: Optional[float]
    matched_sensors: List[str] = None
    id: Optional[int] = None


def sensor_area(width: float, height: float) -> float:
    """Calculate sensor area from dimensions."""
    return width * height


def sensor_size(diag: float, aspect: float) -> Tuple[float, float]:
    """
    Calculate sensor dimensions from diagonal and aspect ratio.

    Args:
        diag: Diagonal in inches
        aspect: Aspect ratio (e.g., 4/3 or 3/2)

    Returns:
        Tuple of (width, height) in mm
    """
    diagmm = diag * 25.4
    h = sqrt(diagmm**2 / (aspect**2 + 1))
    w = aspect * h
    return w, h


def sensor_size_from_type(
    typ: Optional[str], use_table: bool
) -> Optional[Tuple[float, float]]:
    """
    Get sensor size from type designation.

    Args:
        typ: Type designation
        use_table: Whether to use the TYPE_SIZE table

    Returns:
        Diagonal sensor size in mm
    """
    if not typ:
        return None

    if use_table and typ in TYPE_SIZE:
        return TYPE_SIZE[typ]

    if typ.startswith("1/"):
        diag = 1 / float(typ[2:])
        size = sensor_size(diag, 4 / 3)
        return size

    return None


def pixel_pitch(area: float, mpix: float) -> float:
    """
    Calculate pixel pitch from sensor area and megapixels.

    Args:
        area: Sensor area in mm^2
        mpix: Megapixels

    Returns:
        Pixel pitch in µm
    """
    return 1000 * sqrt(area / (mpix * 10**6))


def load_sensors_database() -> dict:
    """Load the sensors database from sensors.json."""
    try:
        with open("sensors.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load sensors.json: {e}")
        return {}


def match_sensors(
    width: Optional[float],
    height: Optional[float],
    megapixels: Optional[float],
    sensors_db: dict,
    size_tolerance: float = 2,  # %
    megapixel_tolerance: float = 5,  # %
) -> List[str]:
    """
    Match camera specifications to sensor models.

    Args:
        width: Sensor width in mm
        height: Sensor height in mm
        megapixels: Megapixels
        sensors_db: Sensors database
        size_tolerance: Tolerance for size matching in mm
        megapixel_tolerance: Tolerance for megapixel matching

    Returns:
        List of matching sensor model names
    """
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

    return sorted(matches)


def load_csv() -> Optional[str]:
    """Load the previous CSV."""
    if os.path.exists("camera-data.csv"):
        return Path("camera-data.csv").read_text(encoding="utf-8")

    return None


def parse_existing_csv(csv_content: str) -> List[SpecDerived]:
    """Parse existing CSV content into SpecDerived objects."""
    if not csv_content:
        return []

    specs = []
    lines = csv_content.strip().split("\n")

    if len(lines) < 2:
        return []

    header = lines[0]

    has_id = header.startswith("id,")

    for line in lines[1:]:
        if not line.strip():
            continue

        try:
            values = []
            current_value = ""
            in_quotes = False

            for char in line:
                if char == '"':
                    if in_quotes and current_value and current_value[-1] == '"':
                        current_value = current_value[:-1] + '"'
                    else:
                        in_quotes = not in_quotes
                elif char == "," and not in_quotes:
                    values.append(current_value)
                    current_value = ""
                else:
                    current_value += char
            values.append(current_value)

            if has_id and len(values) >= 10:
                record_id = int(values[0]) if values[0] else None
                name = values[1]
                type_str = values[2] if values[2] else None
                width_str = values[3]
                height_str = values[4]
                area_str = values[5]
                mpix_str = values[6]
                pitch_str = values[7]
                year_str = values[8]
                sensors_str = values[9] if len(values) > 9 else ""
            elif has_id and len(values) >= 9:
                record_id = int(values[0]) if values[0] else None
                name = values[1]
                type_str = values[2] if values[2] else None
                width_str = values[3]
                height_str = values[4]
                area_str = values[5]
                mpix_str = values[6]
                pitch_str = values[7]
                year_str = values[8]
                sensors_str = ""
            elif not has_id and len(values) >= 9:
                record_id = None
                name = values[0]
                type_str = values[1] if values[1] else None
                width_str = values[2]
                height_str = values[3]
                area_str = values[4]
                mpix_str = values[5]
                pitch_str = values[6]
                year_str = values[7]
                sensors_str = values[8] if len(values) > 8 else ""
            elif not has_id and len(values) >= 8:
                record_id = None
                name = values[0]
                type_str = values[1] if values[1] else None
                width_str = values[2]
                height_str = values[3]
                area_str = values[4]
                mpix_str = values[5]
                pitch_str = values[6]
                year_str = values[7]
                sensors_str = ""
            else:
                continue

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

            spec = Spec(name, type_str, size, pitch, mpix, year)
            derived = SpecDerived(spec, size, area, pitch, matched_sensors, record_id)
            specs.append(derived)

        except Exception as e:
            print(f"Error parsing CSV line: {line[:50]}... - {e}")
            continue

    return specs


def create_camera_key(spec: Spec) -> str:
    """Create a unique key for a camera spec to identify duplicates."""
    return f"{spec.name.lower().strip()}-{spec.year}"


def merge_camera_data(
    new_specs: List[SpecDerived], existing_specs: List[SpecDerived]
) -> List[SpecDerived]:
    """Merge new camera data with existing data, preserving removed cameras."""
    print(
        f"Merging {len(new_specs)} new records with {len(existing_specs)} existing records"
    )

    # Load sensors database for re-matching
    sensors_db = load_sensors_database()

    existing_by_key = {}
    for spec in existing_specs:
        key = create_camera_key(spec.spec)
        existing_by_key[key] = spec

    found_keys = set()
    merged_specs = []

    for new_spec in new_specs:
        key = create_camera_key(new_spec.spec)
        found_keys.add(key)

        if key in existing_by_key:
            existing_spec = existing_by_key[key]
            new_spec.id = existing_spec.id
            print(f"Updated existing camera: {new_spec.spec.name[:50]}")

        merged_specs.append(new_spec)

    for key, existing_spec in existing_by_key.items():
        if key not in found_keys:
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


def extract_entries(url: str) -> list[str]:
    print(f"Fetching {url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html",
        "Cookie": "blaettern=1000",
        "DNT": "1",
        "Connection": "keep-alive",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    entries = re.findall(
        r'class="row productlist__product.+?'
        r'<div class="productlist__bestpriceoffer">',
        response.text,
        re.DOTALL,
    )
    assert entries, "No entries found"
    print(f"Found {len(entries)} entries")
    return entries


def extract_specs(entries: list[str]) -> list[Spec]:
    """Extract camera specifications from HTML entries."""
    specs = []

    for entry in entries:
        name_match = NAME_RE.search(entry)
        if not name_match:
            continue

        type_match = TYPE_RE.search(entry)
        size_match = SIZE_RE.search(entry)
        pitch_match = PITCH_RE.search(entry)
        mpix_match = MPIX_RE.search(entry)
        year_match = YEAR_RE.search(entry)

        name = html.unescape(name_match.group(1))
        name = " ".join(name.split())

        typ = type_match.group(1) if type_match else None

        if size_match:
            width, height = float(size_match.group(1)), float(size_match.group(2))
            size = (width, height)
        else:
            size = None

        pitch = float(pitch_match.group(1)) if pitch_match else None
        mpix = float(mpix_match.group(1)) if mpix_match else None
        year = int(year_match.group(1)) if year_match else None

        specs.append(Spec(name, typ, size, pitch, mpix, year))

    specs = deduplicate_specs(specs)
    return specs


def deduplicate_specs(specs: list[Spec]) -> list[Spec]:
    """Unify product names and remove duplicates."""
    groups: dict[str, list[Spec]] = defaultdict(list)
    rest = []

    # Group possible identical cameras
    for spec in specs:
        match = EXTRAS_RE.search(spec.name)
        if match:
            unified_name = spec.name[: match.start()]
            groups[unified_name].append(spec)
        else:
            rest.append(spec)

    # Check if grouped cameras have the same sensor specs
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
                Spec(unified_name, ref.type, ref.size, ref.pitch, ref.mpix, year)
            )
        else:
            rest.extend(grouped_specs)

    # Remove product numbers in parentheses at end of name
    def remove_parens(spec: Spec) -> Spec:
        name = spec.name.strip()
        match = PARENS_RE.search(name)
        if match:
            name = name[: match.start()].strip()
        return Spec(name, spec.type, spec.size, spec.pitch, spec.mpix, spec.year)

    rest = list(map(remove_parens, rest))
    return rest


def derive_spec(
    spec: Spec, use_size_table: bool = False, sensors_db: Optional[dict] = None
) -> SpecDerived:
    """Derive additional specifications from base spec."""
    if spec.size is None:
        size = sensor_size_from_type(spec.type, use_size_table)
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


def derive_specs(specs: list[Spec], use_size_table: bool = False) -> list[SpecDerived]:
    """Derive specifications for all cameras."""
    sensors_db = load_sensors_database()
    return [derive_spec(spec, use_size_table, sensors_db) for spec in specs]


def get_fixed() -> list[SpecDerived]:
    """Get fixed-lens camera specifications."""
    entries = extract_entries(FIXED_URL)
    return derive_specs(extract_specs(entries), use_size_table=True)


def get_dslrs() -> list[SpecDerived]:
    """Get DSLR camera specifications."""
    entries = extract_entries(DSLR_URL)
    return derive_specs(extract_specs(entries), use_size_table=False)


def get_mirrorless() -> list[SpecDerived]:
    """Get Mirrorless camera specifications."""
    entries = extract_entries(MIRRORLESS_URL)
    return derive_specs(extract_specs(entries), use_size_table=False)


def get_rangefinder() -> list[SpecDerived]:
    """Get Rangefinder camera specifications."""
    entries = extract_entries(RANGEFINDER_URL)
    return derive_specs(extract_specs(entries), use_size_table=False)


def get_all() -> list[SpecDerived]:
    """Get all camera specifications."""
    return get_fixed() + get_dslrs() + get_mirrorless() + get_rangefinder()


def sorted_by(
    specs: list[SpecDerived], key: str = "pitch", reverse: bool = True
) -> list[SpecDerived]:
    """Sort specifications by given key."""
    key_functions = {
        "pitch": lambda c: c.pitch if c.pitch else -1,
        "area": lambda c: c.area if c.area else -1,
        "mpix": lambda c: c.spec.mpix if c.spec.mpix else -1,
        "name": lambda c: c.spec.name,
    }
    return sorted(specs, key=key_functions[key], reverse=reverse)


def prettyprint(derived: SpecDerived) -> None:
    """Pretty print camera specification to console."""
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


env = Environment(
    loader=FileSystemLoader("templates"), autoescape=select_autoescape(["html", "xml"])
)


def datetimeformat(value, format="%d %b %Y %H:%M:%S UTC"):
    return value.strftime(format)


env.filters["formatdate"] = datetimeformat
env.filters["urlencode"] = quote_plus


def write_csv(specs: list[SpecDerived], output_file: Path) -> None:
    """Write camera specifications to CSV file."""
    print(f"Writing CSV to {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            "id,name,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
            "megapixels,pixel_pitch_um,year,matched_sensors\n"
        )

        for derived in specs:
            spec = derived.spec

            id_str = str(derived.id) if derived.id is not None else ""
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

            name_escaped = spec.name.replace('"', '""')
            if "," in name_escaped or '"' in spec.name:
                name_escaped = f'"{name_escaped}"'

            f.write(
                f"{id_str},{name_escaped},{type_str},{width_str},{height_str},"
                f"{area_str},{mpix_str},{pitch_str},{year_str},{sensors_str}\n"
            )


def render_html(output_dir: Path) -> None:
    """Render all HTML files."""
    print("Loading previous CSV artifact...")
    previous_csv = load_csv()
    existing_specs = parse_existing_csv(previous_csv) if previous_csv else []

    print("Fetching camera data...")
    specs_fixedlens = get_fixed()
    specs_dslr = get_dslrs()
    specs_mirrorless = get_mirrorless()
    specs_rangefinder = get_rangefinder()
    new_specs_all = specs_fixedlens + specs_dslr + specs_mirrorless + specs_rangefinder

    specs_all = merge_camera_data(new_specs_all, existing_specs)

    specs_fixedlens = sorted_by(
        [
            s
            for s in specs_all
            if s.spec.name in {spec.spec.name for spec in specs_fixedlens}
        ],
        "pitch",
    )
    specs_dslr = sorted_by(
        [
            s
            for s in specs_all
            if s.spec.name in {spec.spec.name for spec in specs_dslr}
        ],
        "pitch",
    )
    specs_mirrorless = sorted_by(
        [
            s
            for s in specs_all
            if s.spec.name in {spec.spec.name for spec in specs_mirrorless}
        ],
        "pitch",
    )
    specs_rangefinder = sorted_by(
        [
            s
            for s in specs_all
            if s.spec.name in {spec.spec.name for spec in specs_rangefinder}
        ],
        "pitch",
    )
    specs_all = sorted_by(specs_all, "pitch")

    date = datetime.now(UTC)

    print("Generating HTML files...")

    output_dir.mkdir(exist_ok=True)

    template = env.get_template("pixelpitch.html")

    (output_dir / "fixedlens.html").write_text(
        template.render(
            title="Fixed-lens Cameras",
            specs=specs_fixedlens,
            page="fixedlens",
            date=date,
        ),
        encoding="utf-8",
    )

    (output_dir / "dslr.html").write_text(
        template.render(title="DSLR Cameras", specs=specs_dslr, page="dslr", date=date),
        encoding="utf-8",
    )

    (output_dir / "mirrorless.html").write_text(
        template.render(
            title="Mirrorless Cameras",
            specs=specs_mirrorless,
            page="mirrorless",
            date=date,
        ),
        encoding="utf-8",
    )

    (output_dir / "rangefinder.html").write_text(
        template.render(
            title="Rangefinder Cameras",
            specs=specs_rangefinder,
            page="rangefinder",
            date=date,
        ),
        encoding="utf-8",
    )

    (output_dir / "index.html").write_text(
        template.render(title="All Cameras", specs=specs_all, page="all", date=date),
        encoding="utf-8",
    )

    (output_dir / "about.html").write_text(
        env.get_template("about.html").render(page="about"), encoding="utf-8"
    )

    write_csv(specs_all, output_dir / "camera-data.csv")

    static_seo_files = ["robots.txt"]
    for filename in static_seo_files:
        src_path = Path(filename)
        if src_path.exists():
            (output_dir / filename).write_text(
                src_path.read_text(encoding="utf-8"), encoding="utf-8"
            )

    sitemap_content = open("sitemap.xml", "r", encoding="utf-8").read()
    sitemap_content = sitemap_content.replace("2025-07-15", date.strftime("%Y-%m-%d"))

    (output_dir / "sitemap.xml").write_text(sitemap_content, encoding="utf-8")

    print(f"HTML files written to {output_dir}")
    print(f'CSV file written to {output_dir / "camera-data.csv"}')
    print(f"SEO files copied to {output_dir}")


def main():
    if len(sys.argv) > 1:
        match sys.argv[1]:
            case "html":
                output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dist")
                render_html(output_dir)
            case "list":
                print("Fetching all cameras...")
                specs = get_all()
                specs_sorted = sorted_by(specs, "pitch")
                for spec in specs_sorted:
                    if spec.pitch:
                        prettyprint(spec)
            case "--help" | "-h":
                print("Usage: python pixelpitch.py [command] [args]")
                print("\nCommands:")
                print(
                    "  html [dir]    Generate HTML files "
                    "(default: current directory)"
                )
                print("  list          List all cameras with pixel pitch to console")
                print("  --help, -h    Show this help message")
            case _:
                print(f"Unknown command: {sys.argv[1]}")
                print("Run 'python pixelpitch.py --help' for usage information")
                sys.exit(1)
    else:
        render_html(Path("dist"))


if __name__ == "__main__":
    main()
