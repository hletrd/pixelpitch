"""
Offline tests for source parsers.

These do NOT hit the network. They feed locked-in HTML fixtures from
tests/fixtures/ into each source's parser and assert the extracted
values match published reference data.

Run:
    python -m tests.test_parsers_offline

The test deliberately exits with non-zero on the first failure so it can
serve as a CI gate.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sources import imaging_resource, apotelyt, gsmarena, openmvg

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# --------------------------------------------------------------------------
# Test infra

_failures: list[str] = []


def expect(label: str, got, want, *, tol: float = 0.0):
    if isinstance(want, tuple) and isinstance(got, tuple):
        ok = all(abs(a - b) <= tol for a, b in zip(got, want))
    elif want is None:
        ok = got is None
    elif got is None:
        ok = False
    elif isinstance(want, (int, float)):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    mark = "OK" if ok else "FAIL"
    print(f"  [{mark}] {label}: got={got!r} want={want!r}{' (±'+str(tol)+')' if tol else ''}")
    if not ok:
        _failures.append(label)


def section(name: str):
    print(f"\n== {name} ==")


# --------------------------------------------------------------------------
# Imaging Resource — Sony A7 IV fixture
# Published: 35.9 x 23.9 mm, 33 MP -> 5.12 µm, 2021

def test_imaging_resource():
    section("Imaging Resource: Sony A7 IV")
    body = (FIXTURES / "ir_sony_a7_iv.html").read_text(encoding="utf-8")
    fields = imaging_resource._parse_fields(body)
    expect("has Sensor size field",
           bool(fields.get("Sensor size")), True)
    expect("has Pixel Pitch field",
           bool(fields.get("Approximate Pixel Pitch")), True)
    expect("Effective Megapixels",
           fields.get("Effective Megapixels", "").strip(), "33.0")

    # Body category mapping
    expect("body category",
           imaging_resource._body_category(
               fields.get("Camera Format", ""),
               fields.get("Sensor Format", ""),
               fields.get("Model Name", ""),
           ),
           "mirrorless")


# --------------------------------------------------------------------------
# Apotelyt — Sony A7 IV fixture

def test_apotelyt():
    section("Apotelyt: Sony A7 IV")
    body = (FIXTURES / "apotelyt_sony_a7_iv.html").read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    for m in apotelyt.ROW_RE.finditer(body):
        pair = apotelyt._row_to_pair(m.group(1))
        if pair:
            label, value = pair
            fields.setdefault(label, value)

    expect("Camera Model", fields.get("Camera Model"), "Sony A7 IV")

    sm = apotelyt.SIZE_RE.search(fields.get("Sensor Size", ""))
    width = float(sm.group(1)) if sm else None
    height = float(sm.group(2)) if sm else None
    expect("Sensor width",  width,  35.9, tol=0.1)
    expect("Sensor height", height, 23.9, tol=0.1)

    pm = apotelyt.PITCH_RE.search(fields.get("Pixel Pitch", ""))
    pitch = float(pm.group(1)) if pm else None
    expect("Pixel pitch", pitch, 5.12, tol=0.05)

    mm = apotelyt.MPIX_RE.search(fields.get("Sensor Resolution", ""))
    mpix = float(mm.group(1)) if mm else None
    expect("Megapixels", mpix, 32.7, tol=0.5)

    expect("body category",
           apotelyt._body_category(
               fields.get("Camera Type", ""),
               fields.get("Sensor Format", ""),
               fields.get("Camera Model", ""),
           ),
           "mirrorless")


# --------------------------------------------------------------------------
# GSMArena — Galaxy S25 Ultra fixture
# Reference: 200 MP main, Samsung HP2, 1/1.3", 0.6 µm, 2025

def test_gsmarena():
    section("GSMArena: Galaxy S25 Ultra")
    body = (FIXTURES / "gsmarena_s25_ultra.html").read_text(encoding="utf-8")
    fields = gsmarena._parse_spec_table(body)
    cam = fields.get("Triple") or fields.get("Quad") or fields.get("Main Camera") or ""
    expect("found camera entry", bool(cam), True)

    main = gsmarena._select_main_lens(cam)
    expect("found main lens", bool(main), True)

    spec = gsmarena._phone_to_spec("Samsung Galaxy S25 Ultra", fields)
    if spec is None:
        _failures.append("GSMArena spec build")
        print("  [FAIL] _phone_to_spec returned None")
        return
    expect("category",       spec.category, "smartphone")
    expect("type (1/x.y)",   spec.type,     "1/1.3")
    expect("megapixels",     spec.mpix,     200.0, tol=0.1)
    expect("pixel pitch µm", spec.pitch,    0.6,   tol=0.01)
    expect("size width mm",  spec.size[0] if spec.size else None, 9.84, tol=0.1)
    expect("size height mm", spec.size[1] if spec.size else None, 7.40, tol=0.1)
    expect("year",           spec.year,     2025)


# --------------------------------------------------------------------------
# openMVG — synthesise a small CSV and parse via the CSV parser

def test_openmvg_csv_parser():
    section("openMVG CSV parsing")
    csv_body = (
        "CameraMaker,CameraModel,SensorDescription,SensorWidth(mm),"
        "SensorHeight(mm),SensorWidth(pixels),SensorHeight(pixels)\n"
        'Canon,EOS 5D,"36 x 24 mm",36.0,24.0,4368,2912\n'
        'Sony,Alpha 7 III,"35.6 x 23.8 mm",35.6,23.8,6000,4000\n'
        'Apple,iPhone (1/2.5"),"5.75 x 4.32 mm",5.75,4.32,2592,1944\n'
    )
    # Replace http_get for this test
    saved = openmvg.http_get
    try:
        openmvg.http_get = lambda url, **kw: csv_body  # type: ignore
        specs = openmvg.fetch()
    finally:
        openmvg.http_get = saved

    expect("record count", len(specs), 3)
    by_name = {s.name: s for s in specs}
    expect("Canon EOS 5D size",     by_name["Canon EOS 5D"].size,    (36.0, 24.0), tol=0.01)
    expect("Sony Alpha 7 III size", by_name["Sony Alpha 7 III"].size,(35.6, 23.8), tol=0.01)
    expect("iPhone 1/2.5 size",     by_name["Apple iPhone (1/2.5\")"].size, (5.75, 4.32), tol=0.01)
    expect("EOS 5D MP",         by_name["Canon EOS 5D"].mpix,    12.7,  tol=0.1)
    expect("EOS 5D category",   by_name["Canon EOS 5D"].category,    "mirrorless")
    expect("iPhone category",   by_name["Apple iPhone (1/2.5\")"].category, "fixed")


# --------------------------------------------------------------------------
# Merge logic: feeding multi-source CSV records through merge_camera_data

def test_merge_multi_source():
    section("merge_camera_data multi-source")
    import pixelpitch as pp
    from models import Spec, SpecDerived

    def derive(name, category, size, mpix, year):
        spec = Spec(name=name, category=category, type=None,
                    size=size, pitch=None, mpix=mpix, year=year)
        return pp.derive_spec(spec)

    # Existing master CSV state
    existing = [
        derive("Sony A7 IV",       "mirrorless", (35.9, 23.9), 33.0, 2021),
        derive("Old Camera",       "fixed",      (5.0, 3.7),   10.0, 2010),
    ]
    # Mix of fresh Geizhals + IR + GSMArena rows
    new_specs = [
        # Same name as existing → merge
        derive("Sony A7 IV",          "mirrorless", (35.9, 23.9), 33.0, 2021),
        # New IR-only camera
        derive("Sony A1 II",          "mirrorless", (35.9, 24.0), 50.1, 2024),
        # Smartphone (new category)
        derive("Samsung Galaxy S25 Ultra", "smartphone", (9.84, 7.4), 200.0, 2025),
    ]

    merged = pp.merge_camera_data(new_specs, existing)
    names = sorted(s.spec.name for s in merged)
    expect("includes existing-only", "Old Camera" in names, True)
    expect("includes new IR camera", "Sony A1 II" in names, True)
    expect("includes smartphone",    "Samsung Galaxy S25 Ultra" in names, True)
    expect("Sony A7 IV not duplicated",
           sum(1 for n in names if n == "Sony A7 IV"), 1)


# --------------------------------------------------------------------------
# CSV schema: every source should produce records that pass write_csv

def test_csv_schema():
    section("write_csv schema")
    import tempfile

    import pixelpitch as pp
    from models import Spec

    spec = Spec(name="Test, Cam", category="mirrorless", type=None,
                size=(35.9, 23.9), pitch=5.12, mpix=33.0, year=2021)
    derived = pp.derive_spec(spec)
    derived.id = 0

    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        out_path = Path(f.name)
    try:
        pp.write_csv([derived], out_path)
        text = out_path.read_text(encoding="utf-8")
    finally:
        out_path.unlink(missing_ok=True)

    header = text.splitlines()[0]
    expect("header",
           header,
           "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
           "megapixels,pixel_pitch_um,year,matched_sensors")
    row = text.splitlines()[1]
    expect("name quoted (has comma)",
           '"Test, Cam"' in row, True)
    expect("pitch present", "5.12" in row, True)
    expect("year present",  ",2021," in row, True)


def main():
    test_imaging_resource()
    test_apotelyt()
    test_gsmarena()
    test_openmvg_csv_parser()
    test_merge_multi_source()
    test_csv_schema()

    print("\n" + ("=" * 60))
    if _failures:
        print(f"FAILED: {len(_failures)} check(s)")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
