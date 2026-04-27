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
import unittest.mock
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

    # IR_MPIX_RE should match the number, not a stray dot
    m = imaging_resource.IR_MPIX_RE.search("approx. 24.2")
    expect("IR_MPIX_RE skips stray dot", m.group(1) if m else None, "24.2")
    m2 = imaging_resource.IR_MPIX_RE.search("33.0")
    expect("IR_MPIX_RE plain number", m2.group(1) if m2 else None, "33.0")

    # _body_category with hyphenated "Full-Frame"
    expect("IR body category Full-Frame",
           imaging_resource._body_category("", "Full-Frame", ""),
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

    # _body_category with hyphenated "Full-Frame"
    expect("Apotelyt body category Full-Frame",
           apotelyt._body_category("", "Full-Frame", ""),
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
    # Replace http_get for this test using unittest.mock
    with unittest.mock.patch.object(openmvg, 'http_get', return_value=csv_body):
        specs = openmvg.fetch()

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


# --------------------------------------------------------------------------
# parse_existing_csv: comprehensive branch tests

def test_parse_existing_csv():
    section("parse_existing_csv")
    import pixelpitch as pp
    from models import Spec, SpecDerived

    # has_id=True, 11 columns (happy path with sensors)
    csv1 = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        '0,"Test, Cam",mirrorless,,35.90,23.90,858.61,33.0,5.12,2021,IMX455;IMX451\n'
        "1,Canon R5,dslr,,36.00,24.00,864.00,45.0,4.39,2020,\n"
    )
    parsed = pp.parse_existing_csv(csv1)
    expect("row count (has_id, 11 cols)", len(parsed), 2)
    expect("name with comma", parsed[0].spec.name, "Test, Cam")
    expect("category", parsed[0].spec.category, "mirrorless")
    expect("size", parsed[0].size, (35.9, 23.9), tol=0.01)
    expect("area", parsed[0].area, 858.61, tol=0.01)
    expect("mpix", parsed[0].spec.mpix, 33.0, tol=0.1)
    expect("pitch", parsed[0].pitch, 5.12, tol=0.01)
    expect("year", parsed[0].spec.year, 2021)
    expect("matched_sensors", parsed[0].matched_sensors, ["IMX455", "IMX451"])
    expect("record id", parsed[0].id, 0)
    expect("empty sensors", parsed[1].matched_sensors, [])

    # has_id=True, 10 columns (no sensors column)
    csv2 = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year\n"
        "5,Sony A7 IV,mirrorless,,35.90,23.90,858.61,33.0,5.12,2021\n"
    )
    parsed2 = pp.parse_existing_csv(csv2)
    expect("row count (has_id, 10 cols)", len(parsed2), 1)
    expect("no sensors col", parsed2[0].matched_sensors, [])

    # not has_id, 9 columns (with sensors)
    csv3 = (
        "name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year\n"
        "Nikon Z9,mirrorless,,35.90,23.90,858.61,45.7,4.35,2021\n"
    )
    parsed3 = pp.parse_existing_csv(csv3)
    expect("row count (no_id, 9 cols)", len(parsed3), 1)
    expect("no_id name", parsed3[0].spec.name, "Nikon Z9")
    expect("no_id id is None", parsed3[0].id, None)

    # not has_id, 10 columns (with sensors)
    csv4 = (
        "name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        "Nikon Z9,mirrorless,,35.90,23.90,858.61,45.7,4.35,2021,IMX609\n"
    )
    parsed4 = pp.parse_existing_csv(csv4)
    expect("no_id sensors_str reads correct col", parsed4[0].matched_sensors, ["IMX609"])

    # Empty CSV
    parsed_empty = pp.parse_existing_csv("")
    expect("empty CSV returns []", len(parsed_empty), 0)

    # Header only
    parsed_header = pp.parse_existing_csv("id,name,category\n")
    expect("header-only returns []", len(parsed_header), 0)

    # Short row (padded with empty strings, produces a record with sparse data)
    csv5 = "id,name\n1,Short\n"
    parsed5 = pp.parse_existing_csv(csv5)
    expect("short row padded", len(parsed5), 1)
    expect("short row name", parsed5[0].spec.name, "Short")
    expect("short row empty category", parsed5[0].spec.category, "")


# --------------------------------------------------------------------------
# CSV round-trip test

def test_csv_round_trip():
    section("CSV round-trip (write_csv → parse_existing_csv)")
    import tempfile
    import pixelpitch as pp
    from models import Spec, SpecDerived

    spec1 = Spec(name="Test, Camera", category="mirrorless", type="1/2.3",
                  size=(35.9, 23.9), pitch=5.12, mpix=33.0, year=2021)
    d1 = pp.derive_spec(spec1)
    d1.matched_sensors = ["IMX455", "IMX451"]

    spec2 = Spec(name="Simple Cam", category="dslr", type=None,
                  size=None, pitch=None, mpix=None, year=None)
    d2 = pp.derive_spec(spec2)

    derived_list = [d1, d2]
    for i, d in enumerate(derived_list):
        d.id = i

    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        out_path = Path(f.name)
    try:
        pp.write_csv(derived_list, out_path)
        csv_text = out_path.read_text(encoding="utf-8")
    finally:
        out_path.unlink(missing_ok=True)

    parsed_back = pp.parse_existing_csv(csv_text)
    expect("round-trip row count", len(parsed_back), 2)
    expect("round-trip name with comma", parsed_back[0].spec.name, "Test, Camera")
    expect("round-trip size", parsed_back[0].size, (35.9, 23.9), tol=0.01)
    expect("round-trip mpix", parsed_back[0].spec.mpix, 33.0, tol=0.1)
    expect("round-trip pitch", parsed_back[0].pitch, 5.12, tol=0.01)
    expect("round-trip year", parsed_back[0].spec.year, 2021)
    expect("round-trip sensors", parsed_back[0].matched_sensors, ["IMX455", "IMX451"])
    expect("round-trip type", parsed_back[0].spec.type, "1/2.3")
    expect("round-trip None size", parsed_back[1].size, None)
    expect("round-trip None mpix", parsed_back[1].spec.mpix, None)


# --------------------------------------------------------------------------
# deduplicate_specs

def test_deduplicate_specs():
    section("deduplicate_specs")
    import pixelpitch as pp
    from models import Spec

    # Color variants with same specs → unified
    specs = [
        Spec("Sony A7 IV schwarz", "mirrorless", None, (35.9, 23.9), 5.12, 33.0, 2021),
        Spec("Sony A7 IV silber", "mirrorless", None, (35.9, 23.9), 5.12, 33.0, 2021),
    ]
    deduped = pp.deduplicate_specs(specs)
    expect("color variants unified", len(deduped), 1)
    expect("unified name", deduped[0].name, "Sony A7 IV")

    # Color variants with different specs → kept separate
    specs2 = [
        Spec("Cam schwarz", "fixed", None, (5.0, 3.7), 2.0, 10.0, 2020),
        Spec("Cam silber", "fixed", None, (6.0, 4.5), 3.0, 12.0, 2020),
    ]
    deduped2 = pp.deduplicate_specs(specs2)
    expect("different specs kept separate", len(deduped2), 2)

    # Parenthetical suffixes stripped (using non-EXTRAS parenthetical)
    specs3 = [
        Spec("Canon R5 (Wi-Fi)", "mirrorless", None, (36.0, 24.0), 4.39, 45.0, 2020),
    ]
    deduped3 = pp.deduplicate_specs(specs3)
    expect("parens stripped", deduped3[0].name, "Canon R5")

    # Exact duplicates without EXTRAS → deduplicated
    specs4 = [
        Spec("Nikon Z9", "mirrorless", None, (35.9, 23.9), 4.35, 45.7, 2021),
        Spec("Nikon Z9", "mirrorless", None, (35.9, 23.9), 4.35, 45.7, 2021),
    ]
    deduped4 = pp.deduplicate_specs(specs4)
    expect("exact duplicates removed", len(deduped4), 1)

    # Empty input
    deduped_empty = pp.deduplicate_specs([])
    expect("empty input", len(deduped_empty), 0)

    # Same specs but different categories -> both kept
    specs5 = [
        Spec("Canon R5", "mirrorless", None, (36.0, 24.0), 4.39, 45.0, 2020),
        Spec("Canon R5", "dslr", None, (36.0, 24.0), 4.39, 45.0, 2020),
    ]
    deduped5 = pp.deduplicate_specs(specs5)
    expect("different categories kept separate", len(deduped5), 2)

    # Multi-paren name: only last parenthetical should be stripped
    specs6 = [
        Spec("Canon EOS (R5) (Kit)", "mirrorless", None, (36.0, 24.0), 4.39, 45.0, 2020),
    ]
    deduped6 = pp.deduplicate_specs(specs6)
    expect("multi-paren: inner parens preserved", deduped6[0].name, "Canon EOS (R5)")


# --------------------------------------------------------------------------
# merge_camera_data

def test_merge_camera_data():
    section("merge_camera_data")
    import pixelpitch as pp
    from models import Spec, SpecDerived

    def derive(name, category, size, mpix, year):
        spec = Spec(name=name, category=category, type=None,
                    size=size, pitch=None, mpix=mpix, year=year)
        return pp.derive_spec(spec)

    # Overlapping cameras → update
    existing = [derive("Sony A7 IV", "mirrorless", (35.9, 23.9), 33.0, 2021)]
    new = [derive("Sony A7 IV", "mirrorless", (35.9, 23.9), 33.0, 2021)]
    merged = pp.merge_camera_data(new, existing)
    expect("merge: no duplicate on overlap",
           sum(1 for s in merged if s.spec.name == "Sony A7 IV"), 1)

    # Cameras only in existing → preserved
    existing2 = [
        derive("Old Camera", "fixed", (5.0, 3.7), 10.0, 2010),
        derive("Sony A7 IV", "mirrorless", (35.9, 23.9), 33.0, 2021),
    ]
    new2 = [derive("Sony A7 IV", "mirrorless", (35.9, 23.9), 33.0, 2021)]
    merged2 = pp.merge_camera_data(new2, existing2)
    names2 = [s.spec.name for s in merged2]
    expect("merge: preserves existing-only", "Old Camera" in names2, True)

    # Cameras only in new → added
    existing3 = []
    new3 = [derive("Canon R5", "dslr", (36.0, 24.0), 45.0, 2020)]
    merged3 = pp.merge_camera_data(new3, existing3)
    expect("merge: adds new-only", len(merged3), 1)

    # None years don't create "None" in key
    existing4 = [derive("Cam X", "fixed", (5.0, 3.7), 10.0, None)]
    new4 = [derive("Cam X", "fixed", (5.0, 3.7), 10.0, None)]
    merged4 = pp.merge_camera_data(new4, existing4)
    expect("merge: None year no duplicate",
           sum(1 for s in merged4 if s.spec.name == "Cam X"), 1)


# --------------------------------------------------------------------------
# sensor_size_from_type

def test_sensor_size_from_type():
    section("sensor_size_from_type")
    import pixelpitch as pp

    # Type in lookup table → always returns table value
    result = pp.sensor_size_from_type("1/2.3", use_table=True)
    expect("1/2.3 with table=True", result, (6.17, 4.55), tol=0.01)

    result2 = pp.sensor_size_from_type("1/2.3", use_table=False)
    expect("1/2.3 with table=False (still uses table)", result2, (6.17, 4.55), tol=0.01)

    # Type not in lookup table but starts with "1/" → computed
    result3 = pp.sensor_size_from_type("1/3.1", use_table=True)
    expect("1/3.1 computed (not in table)", result3 is not None, True)
    # Computed value will be approximate
    if result3:
        expect("1/3.1 width > 0", result3[0] > 0, True)
        expect("1/3.1 height > 0", result3[1] > 0, True)

    # None type
    result4 = pp.sensor_size_from_type(None, use_table=True)
    expect("None type returns None", result4, None)

    # Unknown type (not 1/x format, not in table)
    result5 = pp.sensor_size_from_type("APS-C", use_table=True)
    expect("unknown type returns None", result5, None)

    # Phone-format sensor types (merged from gsmarena.PHONE_TYPE_SIZE)
    result6 = pp.sensor_size_from_type("1/1.3", use_table=True)
    expect("1/1.3 measured width", result6[0], 9.84, tol=0.01)
    expect("1/1.3 measured height", result6[1], 7.40, tol=0.01)

    result7 = pp.sensor_size_from_type("1/1.7", use_table=True)
    expect("1/1.7 measured width", result7[0], 7.60, tol=0.01)
    expect("1/1.7 measured height", result7[1], 5.70, tol=0.01)

    result8 = pp.sensor_size_from_type("1/2.8", use_table=True)
    expect("1/2.8 measured width", result8[0], 5.12, tol=0.01)
    expect("1/2.8 measured height", result8[1], 3.84, tol=0.01)


# --------------------------------------------------------------------------
# pixel_pitch

def test_pixel_pitch():
    section("pixel_pitch")
    import pixelpitch as pp

    # Full-frame 33MP: 864mm2 / 33e6 → sqrt * 1000
    pitch = pp.pixel_pitch(864.0, 33.0)
    expect("A7 IV pitch", pitch, 5.12, tol=0.05)

    # Small sensor: 27.94mm2 / 0.8MP
    pitch2 = pp.pixel_pitch(27.94, 0.8)
    expect("small sensor pitch", pitch2, 5.91, tol=0.1)

    # Edge case: 1MP on 1mm2
    pitch3 = pp.pixel_pitch(1.0, 1.0)
    expect("1mm2 1MP pitch", pitch3, 1.0, tol=0.01)


# --------------------------------------------------------------------------
# match_sensors

def test_match_sensors():
    section("match_sensors")
    import pixelpitch as pp

    sensors_db = {
        "IMX455": {"sensor_width_mm": 36.0, "sensor_height_mm": 24.0, "megapixels": [61.2, 61.0]},
        "IMX410": {"sensor_width_mm": 35.9, "sensor_height_mm": 23.9, "megapixels": [24.2, 24.6]},
    }

    # Width+height+mpix match
    matches = pp.match_sensors(36.0, 24.0, 61.0, sensors_db)
    expect("match with mpix", "IMX455" in matches, True)

    # Width+height match, None mpix → should still match (F10 fix)
    matches2 = pp.match_sensors(36.0, 24.0, None, sensors_db)
    expect("match with None mpix", "IMX455" in matches2, True)

    # No match
    matches3 = pp.match_sensors(10.0, 8.0, 20.0, sensors_db)
    expect("no match", len(matches3), 0)

    # Empty db
    matches4 = pp.match_sensors(36.0, 24.0, 61.0, {})
    expect("empty db", len(matches4), 0)


def test_load_sensors_database():
    section("load_sensors_database error handling")
    import pixelpitch as pp

    # PermissionError (subclass of OSError) should return {} gracefully
    with unittest.mock.patch("builtins.open", side_effect=PermissionError("Permission denied")):
        result = pp.load_sensors_database()
    expect("PermissionError returns {}", result, {})

    # FileNotFoundError should return {} gracefully
    with unittest.mock.patch("builtins.open", side_effect=FileNotFoundError("No such file")):
        result2 = pp.load_sensors_database()
    expect("FileNotFoundError returns {}", result2, {})

    # json.JSONDecodeError should return {} gracefully
    with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data="not json")):
        result3 = pp.load_sensors_database()
    expect("JSONDecodeError returns {}", result3, {})


# --------------------------------------------------------------------------
# CineD FORMAT_TO_MM completeness: every regex-capturable format must have
# a corresponding lookup-table entry.

def test_cined_format_coverage():
    section("CineD FORMAT_TO_MM completeness")
    from sources.cined import FORMAT_TO_MM
    import re

    # Regex alternation groups from _parse_camera_page
    fmt_re = re.compile(
        r"(Full Frame|Super 35(?:\s*mm)?|APS-C|Micro Four Thirds|Four Thirds|"
        r'1"|1-inch|2/3"|Medium Format)',
        re.IGNORECASE,
    )

    # Extract all possible match groups by testing known strings
    test_strings = [
        "Full Frame", "Super 35", "Super 35 mm", "APS-C",
        "Micro Four Thirds", "Four Thirds", '1"', "1-inch",
        '2/3"', "Medium Format",
    ]
    for fmt_str in test_strings:
        m = fmt_re.search(fmt_str)
        if m:
            key = m.group(1).lower()
            val = FORMAT_TO_MM.get(key)
            expect(f"FORMAT_TO_MM[{key!r}]", val is not None, True)


def main():
    test_imaging_resource()
    test_apotelyt()
    test_gsmarena()
    test_openmvg_csv_parser()
    test_merge_multi_source()
    test_csv_schema()
    test_parse_existing_csv()
    test_csv_round_trip()
    test_deduplicate_specs()
    test_merge_camera_data()
    test_sensor_size_from_type()
    test_pixel_pitch()
    test_match_sensors()
    test_load_sensors_database()
    test_cined_format_coverage()

    print("\n" + ("=" * 60))
    if _failures:
        print(f"FAILED: {len(_failures)} check(s)")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
