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

    # _body_category name-based action cam detection
    expect("IR body category GoPro name",
           imaging_resource._body_category("", "", "GoPro Hero 12"), "actioncam")
    expect("IR body category Insta360 name",
           imaging_resource._body_category("", "", "Insta360 X4"), "actioncam")
    expect("IR body category Osmo Action name",
           imaging_resource._body_category("", "", "DJI Osmo Action 4"), "actioncam")

    # _body_category name-based camcorder detection
    expect("IR body category Handycam name",
           imaging_resource._body_category("", "", "Sony Handycam FDR-AX43"), "camcorder")

    # _body_category sensor format fallbacks
    expect("IR body category APS-C sensor format",
           imaging_resource._body_category("", "APS-C", ""), "mirrorless")
    expect("IR body category Micro Four Thirds sensor format",
           imaging_resource._body_category("", "Micro Four Thirds", ""), "mirrorless")
    expect("IR body category Medium Format sensor format",
           imaging_resource._body_category("", "Medium Format", ""), "mirrorless")

    # _body_category final fallback to "fixed" for small sensor formats
    expect("IR body category 1/2.3 sensor format",
           imaging_resource._body_category("", "1/2.3", ""), "fixed")

    # _parse_camera_name with modern spec URL (review/specifications/)
    name_modern = imaging_resource._parse_camera_name(
        {"Model Name": "Sony Alpha ILCE-A7 IV"},
        "https://www.imaging-resource.com/cameras/sony-a7-iv-review/specifications/"
    )
    expect("IR Sony name from modern spec URL",
           name_modern, "Sony A7 IV")

    # _parse_camera_name with legacy spec URL (slug-specifications/)
    name_legacy = imaging_resource._parse_camera_name(
        {"Model Name": "Sony Alpha ILCZV-E10"},
        "https://www.imaging-resource.com/cameras/sony-zv-e10-specifications/"
    )
    expect("IR Sony name from legacy spec URL",
           name_legacy, "Sony ZV-E10")

    # _parse_camera_name fallback with empty Model Name (modern URL)
    name_fb_modern = imaging_resource._parse_camera_name(
        {"Model Name": ""},
        "https://www.imaging-resource.com/cameras/nikon-z9-review/specifications/"
    )
    expect("IR fallback name from modern spec URL",
           name_fb_modern, "Nikon Z9")

    # _parse_camera_name fallback with empty Model Name (legacy URL)
    name_fb_legacy = imaging_resource._parse_camera_name(
        {"Model Name": ""},
        "https://www.imaging-resource.com/cameras/nikon-z9-specifications/"
    )
    expect("IR fallback name from legacy spec URL",
           name_fb_legacy, "Nikon Z9")

    # _parse_camera_name fallback with Sony URL and empty Model Name
    # Should apply Sony-specific normalizations (Roman numerals, ZV replacement)
    name_sony_fb_modern = imaging_resource._parse_camera_name(
        {"Model Name": ""},
        "https://www.imaging-resource.com/cameras/sony-a7-iv-review/specifications/"
    )
    expect("IR Sony fallback name from modern URL",
           name_sony_fb_modern, "Sony A7 IV")

    name_sony_fb_legacy = imaging_resource._parse_camera_name(
        {"Model Name": ""},
        "https://www.imaging-resource.com/cameras/sony-a7-iv-specifications/"
    )
    expect("IR Sony fallback name from legacy URL",
           name_sony_fb_legacy, "Sony A7 IV")

    name_sony_fb_zv = imaging_resource._parse_camera_name(
        {"Model Name": ""},
        "https://www.imaging-resource.com/cameras/sony-zv-e10-review/specifications/"
    )
    expect("IR Sony fallback ZV name",
           name_sony_fb_zv, "Sony ZV-E10")

    # Sony FX series naming — .title() converts "fx3" to "Fx3";
    # the normalizer must correct this to "FX3"
    name_fx3 = imaging_resource._parse_camera_name(
        {"Model Name": "Sony FX3"},
        "https://www.imaging-resource.com/cameras/sony-fx3-review/specifications/"
    )
    expect("IR Sony FX3 name", name_fx3, "Sony FX3")

    name_fx30 = imaging_resource._parse_camera_name(
        {"Model Name": "Sony FX30"},
        "https://www.imaging-resource.com/cameras/sony-fx30-review/specifications/"
    )
    expect("IR Sony FX30 name", name_fx30, "Sony FX30")

    # FX fallback from URL with empty Model Name
    name_fx_fb = imaging_resource._parse_camera_name(
        {"Model Name": ""},
        "https://www.imaging-resource.com/cameras/sony-fx6-review/specifications/"
    )
    expect("IR Sony FX6 fallback name", name_fx_fb, "Sony FX6")

    # Sony RX series naming — .title() converts "rx" to "Rx";
    # the normalizer must correct this to "RX"
    name_rx100 = imaging_resource._parse_camera_name(
        {"Model Name": "Sony RX100 VII"},
        "https://www.imaging-resource.com/cameras/sony-rx100-vii-review/specifications/"
    )
    expect("IR Sony RX100 VII name", name_rx100, "Sony RX100 VII")

    name_rx10 = imaging_resource._parse_camera_name(
        {"Model Name": "Sony RX10 IV"},
        "https://www.imaging-resource.com/cameras/sony-rx10-iv-review/specifications/"
    )
    expect("IR Sony RX10 IV name", name_rx10, "Sony RX10 IV")

    # Sony DSC series naming — .title() converts "dsc" to "Dsc"
    name_dsc = imaging_resource._parse_camera_name(
        {"Model Name": "Sony DSC-HX400"},
        "https://www.imaging-resource.com/cameras/sony-dsc-hx400-review/specifications/"
    )
    expect("IR Sony DSC-HX400 name", name_dsc, "Sony DSC HX400")

    # Sony HX series — within DSC name
    name_hx = imaging_resource._parse_camera_name(
        {"Model Name": "Sony DSC-WX350"},
        "https://www.imaging-resource.com/cameras/sony-dsc-wx350-review/specifications/"
    )
    expect("IR Sony DSC-WX350 name", name_hx, "Sony DSC WX350")

    # RX fallback from URL with empty Model Name
    name_rx_fb = imaging_resource._parse_camera_name(
        {"Model Name": ""},
        "https://www.imaging-resource.com/cameras/sony-rx1r-ii-review/specifications/"
    )
    expect("IR Sony RX1R II fallback name", name_rx_fb, "Sony RX1R II")


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
    # spec.size is None because GSMArena only provides the fractional-inch
    # type, not measured dimensions. The type-derived dimensions are in
    # derived.size (computed by derive_spec from spec.type).
    expect("spec.size is None (type-derived, not measured)", spec.size, None)
    expect("year",           spec.year,     2025)

    # Verify derive_spec computes derived.size from spec.type
    import pixelpitch as pp
    derived = pp.derive_spec(spec)
    expect("derived.size from type 1/1.3 width",
           derived.size[0] if derived.size else None, 9.84, tol=0.1)
    expect("derived.size from type 1/1.3 height",
           derived.size[1] if derived.size else None, 7.40, tol=0.1)


def test_gsmarena_unicode_quotes():
    """Verify TYPE_FRACTIONAL_RE (formerly SENSOR_FORMAT_RE) matches Unicode curly quotes."""
    section("GSMArena TYPE_FRACTIONAL_RE Unicode quotes")
    from sources import TYPE_FRACTIONAL_RE
    # ASCII double-quote
    m1 = TYPE_FRACTIONAL_RE.search('1/1.3"')
    expect("ASCII quote match", m1.group(1) if m1 else None, "1/1.3")
    # Unicode right double quotation mark (U+2033)
    m2 = TYPE_FRACTIONAL_RE.search('1/1.3″')
    expect("Unicode quote match", m2.group(1) if m2 else None, "1/1.3")
    # "-inch" suffix
    m3 = TYPE_FRACTIONAL_RE.search('1/2.3-inch')
    expect("inch suffix match", m3.group(1) if m3 else None, "1/2.3")
    # No match for bare number without suffix
    m4 = TYPE_FRACTIONAL_RE.search('1/2.3 other')
    expect("no suffix no match", m4 is None, True)
    # Space+inch suffix ("1/2.3 inch")
    m5 = TYPE_FRACTIONAL_RE.search('1/2.3 inch')
    expect("space+inch suffix match", m5.group(1) if m5 else None, "1/2.3")
    # "inch" without space still matches (subsumed by \s*inch)
    m6 = TYPE_FRACTIONAL_RE.search('1/2.3inch')
    expect("no-space inch suffix match", m6.group(1) if m6 else None, "1/2.3")


def test_mpix_re_format_handling():
    """Verify centralized MPIX_RE matches 'Megapixel', 'MP', and 'Mega pixels'."""
    section("MPIX_RE format handling")
    from sources import MPIX_RE

    # Classic "Megapixel" — must still work
    m1 = MPIX_RE.search("33.0 Megapixel")
    expect("MPIX_RE matches Megapixel", m1.group(1) if m1 else None, "33.0")

    # "MP" abbreviation
    m2 = MPIX_RE.search("33.0 MP")
    expect("MPIX_RE matches MP", m2.group(1) if m2 else None, "33.0")

    # "Mega pixels" with space
    m3 = MPIX_RE.search("33.0 Mega pixels")
    expect("MPIX_RE matches Mega pixels", m3.group(1) if m3 else None, "33.0")

    # "Megapixels" plural
    m4 = MPIX_RE.search("32.7 Megapixels")
    expect("MPIX_RE matches Megapixels", m4.group(1) if m4 else None, "32.7")

    # "effective Megapixels" prefix
    m5 = MPIX_RE.search("effective 45.7 Megapixels")
    expect("MPIX_RE matches effective prefix", m5.group(1) if m5 else None, "45.7")

    # Case-insensitive: "mp" lowercase
    m6 = MPIX_RE.search("24.2 mp")
    expect("MPIX_RE matches lowercase mp", m6.group(1) if m6 else None, "24.2")


# --------------------------------------------------------------------------
# openMVG — synthesise a small CSV and parse via the CSV parser

def test_openmvg_csv_parser():
    section("openMVG CSV parsing")
    csv_body = (
        "CameraMaker,CameraModel,SensorDescription,SensorWidth(mm),"
        "SensorHeight(mm),SensorWidth(pixels),SensorHeight(pixels)\n"
        'Canon,EOS 5D,"36 x 24 mm",36.0,24.0,4368,2912\n'
        'Canon,EOS 250D,"22.3 x 14.9 mm",22.3,14.9,6032,4016\n'
        'Samsung,NX300,"23.5 x 15.7 mm",23.5,15.7,5472,3648\n'
        'Sigma,SD14,"20.7 x 13.8 mm",20.7,13.8,2652,1768\n'
        'Sony,Alpha 7 III,"35.6 x 23.8 mm",35.6,23.8,6000,4000\n'
        'Apple,iPhone (1/2.5"),"5.75 x 4.32 mm",5.75,4.32,2592,1944\n'
        'Pentax,K3,"23.5 x 15.6 mm",23.5,15.6,6016,4000\n'
        'Pentax,645Z,"43.8 x 32.9 mm",43.8,32.9,8256,6192\n'
        'Pentax,KP,"23.5 x 15.6 mm",23.5,15.6,6016,4000\n'
        'Pentax,KF,"23.5 x 15.6 mm",23.5,15.6,6016,4000\n'
        'Pentax,K-r,"23.5 x 15.6 mm",23.5,15.6,4928,3280\n'
        'Pentax,K-x,"23.5 x 15.6 mm",23.5,15.6,4288,2848\n'
        'Nikon,Df,"36.0 x 23.9 mm",36.0,23.9,4928,3280\n'
    )
    # Replace http_get for this test using unittest.mock
    with unittest.mock.patch.object(openmvg, 'http_get', return_value=csv_body):
        specs = openmvg.fetch()

    expect("record count", len(specs), 13)
    by_name = {s.name: s for s in specs}
    expect("Canon EOS 5D size",     by_name["Canon EOS 5D"].size,    (36.0, 24.0), tol=0.01)
    expect("Sony Alpha 7 III size", by_name["Sony Alpha 7 III"].size,(35.6, 23.8), tol=0.01)
    expect("iPhone 1/2.5 size",     by_name["Apple iPhone (1/2.5\")"].size, (5.75, 4.32), tol=0.01)
    expect("EOS 5D MP",         by_name["Canon EOS 5D"].mpix,    12.7,  tol=0.1)
    expect("EOS 5D category",   by_name["Canon EOS 5D"].category,    "dslr")
    expect("Sony Alpha 7 III category", by_name["Sony Alpha 7 III"].category, "mirrorless")
    expect("iPhone category",   by_name["Apple iPhone (1/2.5\")"].category, "fixed")
    # DSLR regex correctness tests
    expect("EOS 250D category (xxxD DSLR)", by_name["Canon EOS 250D"].category, "dslr")
    expect("Samsung NX300 category (mirrorless, NOT DSLR)", by_name["Samsung NX300"].category, "mirrorless")
    expect("Sigma SD14 category (2-digit DSLR)", by_name["Sigma SD14"].category, "dslr")
    # Pentax DSLR regex tests
    expect("Pentax K3 category (no-hyphen DSLR)", by_name["Pentax K3"].category, "dslr")
    expect("Pentax 645Z category (medium-format DSLR)", by_name["Pentax 645Z"].category, "dslr")
    expect("Pentax KP category (letter-suffix DSLR)", by_name["Pentax KP"].category, "dslr")
    expect("Pentax KF category (letter-suffix DSLR)", by_name["Pentax KF"].category, "dslr")
    expect("Pentax K-r category (hyphen+letter DSLR)", by_name["Pentax K-r"].category, "dslr")
    expect("Pentax K-x category (hyphen+letter DSLR)", by_name["Pentax K-x"].category, "dslr")
    # Nikon letter-suffix DSLR test
    expect("Nikon Df category (letter-suffix DSLR)", by_name["Nikon Df"].category, "dslr")


def test_openmvg_bom():
    """Verify openMVG fetch handles BOM-prefixed CSV correctly."""
    section("openMVG BOM handling")
    # Same CSV as above but with UTF-8 BOM prefix
    csv_body = (
        "﻿CameraMaker,CameraModel,SensorDescription,SensorWidth(mm),"
        "SensorHeight(mm),SensorWidth(pixels),SensorHeight(pixels)\n"
        'Canon,EOS 5D,"36 x 24 mm",36.0,24.0,4368,2912\n'
    )
    with unittest.mock.patch.object(openmvg, 'http_get', return_value=csv_body):
        specs = openmvg.fetch()

    expect("BOM: record count", len(specs), 1)
    if specs:
        expect("BOM: name parsed", specs[0].name, "Canon EOS 5D")
        expect("BOM: size", specs[0].size, (36.0, 24.0), tol=0.01)
        expect("BOM: category", specs[0].category, "dslr")


def test_openmvg_negative_dimensions():
    """Verify openMVG fetch rejects negative sensor dimensions."""
    section("openMVG negative dimension validation")
    csv_body = (
        "CameraMaker,CameraModel,SensorDescription,SensorWidth(mm),"
        "SensorHeight(mm),SensorWidth(pixels),SensorHeight(pixels)\n"
        'Canon,EOS Neg,"-1.0 x -1.0 mm",-1.0,-1.0,100,100\n'
    )
    with unittest.mock.patch.object(openmvg, 'http_get', return_value=csv_body):
        specs = openmvg.fetch()
    if specs:
        expect("negative dimensions rejected", specs[0].size, None)
    else:
        expect("negative dimensions: 0 records", True, True)

    # Zero dimensions should also be rejected (bool(0.0) is False)
    csv_zero = (
        "CameraMaker,CameraModel,SensorDescription,SensorWidth(mm),"
        "SensorHeight(mm),SensorWidth(pixels),SensorHeight(pixels)\n"
        'Canon,EOS Zero,"0 x 0 mm",0.0,0.0,0,0\n'
    )
    with unittest.mock.patch.object(openmvg, 'http_get', return_value=csv_zero):
        specs_zero = openmvg.fetch()
    if specs_zero:
        expect("zero dimensions rejected", specs_zero[0].size, None)
    else:
        expect("zero dimensions: 0 records", True, True)

    # Negative pixel dimensions must produce mpix=None (not positive mpix
    # from the product of two negatives)
    csv_neg_pixels = (
        "CameraMaker,CameraModel,SensorDescription,SensorWidth(mm),"
        "SensorHeight(mm),SensorWidth(pixels),SensorHeight(pixels)\n"
        'Canon,EOS NegPixels,"1 x 1 mm",1.0,1.0,-100,-200\n'
    )
    with unittest.mock.patch.object(openmvg, 'http_get', return_value=csv_neg_pixels):
        specs_neg_px = openmvg.fetch()
    if specs_neg_px:
        expect("negative pixel dimensions: mpix is None",
               specs_neg_px[0].mpix, None)
    else:
        expect("negative pixel dimensions: 0 records", True, True)


# --------------------------------------------------------------------------
# sorted_by: 0.0 values must sort at 0.0, not -1

def test_sorted_by_zero_values():
    """Verify sorted_by treats invalid pitch (0.0, None) consistently."""
    section("sorted_by invalid pitch handling")
    import pixelpitch as pp
    from models import Spec

    def derive(name, category, size, mpix, year, pitch_val=None):
        spec = Spec(name=name, category=category, type=None,
                    size=size, pitch=pitch_val, mpix=mpix, year=year)
        return pp.derive_spec(spec)

    # Camera with pitch=0.0 and mpix=0.0 — derive_spec produces None (invalid)
    cam_zero = derive("Zero Pitch", "fixed", (5.0, 3.7), 0.0, 2020, pitch_val=0.0)
    cam_normal = derive("Normal Pitch", "fixed", (35.9, 23.9), 33.0, 2021, pitch_val=5.12)
    cam_none = derive("No Pitch", "fixed", None, None, 2019)

    sorted_list = pp.sorted_by([cam_normal, cam_zero, cam_none], "pitch")
    # Both cam_zero and cam_none have derived.pitch=None, so they sort at -1
    # Sorted descending by pitch: Normal (5.12) > Zero (None=-1) = No Pitch (None=-1)
    expect("sorted_by: normal-pitch camera at index 0",
           sorted_list[0].spec.name, "Normal Pitch")
    # cam_zero and cam_none both have pitch=None, so their relative order
    # depends on the stable sort — just verify both are at indices 1 and 2
    names_at_end = {sorted_list[1].spec.name, sorted_list[2].spec.name}
    expect("sorted_by: zero and none-pitch cameras at end",
           names_at_end == {"Zero Pitch", "No Pitch"}, True)


# --------------------------------------------------------------------------
# Template rendering: 0.0 pitch/mpix are physically impossible and must
# render as "unknown" (not as numeric values), consistent with JS
# isInvalidData treating pitch===0 as invalid.

def test_template_zero_pitch_rendering():
    """Verify pixelpitch.html template renders 0.0 pitch/mpix as 'unknown'."""
    section("template 0.0 value rendering")
    import pixelpitch as pp
    from models import Spec

    spec = Spec(name="Zero Cam", category="fixed", type=None,
                size=(5.0, 3.7), pitch=0.0, mpix=0.0, year=2020)
    d = pp.derive_spec(spec)
    d.id = 0

    from datetime import datetime, timezone
    date = datetime.now(timezone.utc)

    html = pp._get_env().get_template("pixelpitch.html").render(
        title="Test", specs=[d], page="fixed", date=date,
    )

    # 0.0 mpix is physically impossible — must render as "unknown", not "0.0 MP"
    expect("template: 0.0 mpix renders as unknown",
           "0.0 MP" not in html, True)
    expect("template: 0.0 mpix shows unknown text",
           "unknown" in html, True)

    # 0.0 pitch is physically impossible — must render as "unknown", not "0.0 µm"
    expect("template: 0.0 pitch renders as unknown",
           "0.0 µm" not in html and "0.0 µm" not in html, True)


def test_template_negative_pitch_rendering():
    """Verify pixelpitch.html template renders negative pitch/mpix as 'unknown'."""
    section("template negative value rendering")
    import pixelpitch as pp
    from models import Spec

    spec = Spec(name="Neg Cam", category="fixed", type=None,
                size=(5.0, 3.7), pitch=-1.0, mpix=-10.0, year=2020)
    d = pp.derive_spec(spec)
    d.id = 0

    from datetime import datetime, timezone
    date = datetime.now(timezone.utc)

    html = pp._get_env().get_template("pixelpitch.html").render(
        title="Test", specs=[d], page="fixed", date=date,
    )

    # Negative mpix is physically impossible — must render as "unknown", not "-10.0 MP"
    expect("template: negative mpix renders as unknown",
           "-10.0 MP" not in html, True)
    expect("template: negative mpix shows unknown text",
           "unknown" in html, True)

    # Negative pitch is physically impossible — must render as "unknown", not "-1.0 µm"
    expect("template: negative pitch renders as unknown",
           "-1.0 µm" not in html, True)


def test_parse_existing_csv_negative_values():
    """Verify parse_existing_csv rejects negative physical quantities."""
    section("parse_existing_csv negative value rejection")
    import pixelpitch as pp

    csv_neg = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        "0,Test,mirrorless,,-5.00,-3.00,-15.00,-10.0,-1.00,2021,\n"
    )
    parsed = pp.parse_existing_csv(csv_neg)
    expect("negative CSV: row count", len(parsed), 1)
    expect("negative CSV: size is None", parsed[0].size, None)
    expect("negative CSV: area is None", parsed[0].area, None)
    expect("negative CSV: mpix is None", parsed[0].spec.mpix, None)
    expect("negative CSV: pitch is None", parsed[0].pitch, None)


# --------------------------------------------------------------------------
# Merge logic: feeding multi-source CSV records through merge_camera_data

def test_merge_multi_source():
    section("merge_camera_data multi-source")
    import pixelpitch as pp
    from models import Spec

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

    # Type field with leading/trailing whitespace — should be stripped
    csv6 = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        '6,Test Cam,mirrorless, 1/2.3 ,6.17,4.55,28.07,12.0,1.55,2020,\n'
    )
    parsed6 = pp.parse_existing_csv(csv6)
    expect("whitespace type stripped", parsed6[0].spec.type, "1/2.3")

    # Name field with leading/trailing whitespace — should be stripped
    csv7 = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        '7, Sony A7 IV ,mirrorless,,35.90,23.90,858.61,33.0,5.12,2021,\n'
    )
    parsed7 = pp.parse_existing_csv(csv7)
    expect("whitespace name stripped", parsed7[0].spec.name, "Sony A7 IV")

    # UTF-8 BOM prefix — should be stripped so schema detection works
    csv8 = (
        "﻿id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        '8,Sony A7 IV,mirrorless,,35.90,23.90,858.61,33.0,5.12,2021,\n'
    )
    parsed8 = pp.parse_existing_csv(csv8)
    expect("BOM: row count", len(parsed8), 1)
    expect("BOM: name parsed", parsed8[0].spec.name, "Sony A7 IV")
    expect("BOM: record id", parsed8[0].id, 8)
    expect("BOM: category", parsed8[0].spec.category, "mirrorless")

    # Year validation: year=0 should be rejected (treated as None)
    csv_y0 = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        "0,Test,mirrorless,,36.00,24.00,864.00,45.0,5.00,0,\n"
    )
    parsed_y0 = pp.parse_existing_csv(csv_y0)
    expect("year=0 rejected", parsed_y0[0].spec.year, None)

    # Year validation: negative year should be rejected
    csv_neg = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        "0,Test,mirrorless,,36.00,24.00,864.00,45.0,5.00,-1,\n"
    )
    parsed_neg = pp.parse_existing_csv(csv_neg)
    expect("year=-1 rejected", parsed_neg[0].spec.year, None)

    # Year validation: year outside range should be rejected
    csv_far = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        "0,Test,mirrorless,,36.00,24.00,864.00,45.0,5.00,99999,\n"
    )
    parsed_far = pp.parse_existing_csv(csv_far)
    expect("year=99999 rejected", parsed_far[0].spec.year, None)

    # Year validation: valid year still accepted
    csv_valid = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        "0,Test,mirrorless,,36.00,24.00,864.00,45.0,5.00,2021,\n"
    )
    parsed_valid = pp.parse_existing_csv(csv_valid)
    expect("year=2021 accepted", parsed_valid[0].spec.year, 2021)

    # matched_sensors: leading/trailing/doubled semicolons must not produce empty strings
    csv_semicolons = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        "0,Test,mirrorless,,36.00,24.00,864.00,45.0,5.00,2021,;IMX455;;IMX410;\n"
    )
    parsed_semi = pp.parse_existing_csv(csv_semicolons)
    expect("matched_sensors: no empty strings from semicolons",
           '' not in parsed_semi[0].matched_sensors, True)
    expect("matched_sensors: correct entries from semicolons",
           parsed_semi[0].matched_sensors, ["IMX455", "IMX410"])

    # NaN values in CSV should be treated as None
    csv_nan = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        "0,Test,mirrorless,,nan,nan,nan,nan,nan,2021,\n"
    )
    parsed_nan = pp.parse_existing_csv(csv_nan)
    expect("NaN CSV: row count", len(parsed_nan), 1)
    expect("NaN CSV: size is None", parsed_nan[0].size, None)
    expect("NaN CSV: area is None", parsed_nan[0].area, None)
    expect("NaN CSV: mpix is None", parsed_nan[0].spec.mpix, None)
    expect("NaN CSV: pitch is None", parsed_nan[0].pitch, None)

    # inf values in CSV should be treated as None
    csv_inf = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        "0,Test,mirrorless,,inf,inf,inf,inf,inf,2021,\n"
    )
    parsed_inf = pp.parse_existing_csv(csv_inf)
    expect("inf CSV: row count", len(parsed_inf), 1)
    expect("inf CSV: size is None", parsed_inf[0].size, None)
    expect("inf CSV: area is None", parsed_inf[0].area, None)
    expect("inf CSV: mpix is None", parsed_inf[0].spec.mpix, None)
    expect("inf CSV: pitch is None", parsed_inf[0].pitch, None)


# --------------------------------------------------------------------------
# CSV round-trip test

def test_csv_round_trip():
    section("CSV round-trip (write_csv → parse_existing_csv)")
    import tempfile
    import pixelpitch as pp
    from models import Spec

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

    # 0.0 value preservation: float fields that are 0.0 must survive
    # the CSV round-trip (not silently dropped by falsy checks).
    spec_zero = Spec(name="Zero MP Cam", category="fixed", type=None,
                     size=(5.0, 3.7), pitch=0.0, mpix=0.0, year=2020)
    d_zero = pp.derive_spec(spec_zero)
    d_zero.id = 2

    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        out_path_zero = Path(f.name)
    try:
        pp.write_csv([d_zero], out_path_zero)
        csv_zero_text = out_path_zero.read_text(encoding="utf-8")
    finally:
        out_path_zero.unlink(missing_ok=True)

    parsed_zero = pp.parse_existing_csv(csv_zero_text)
    # 0.0 mpix/pitch are physically impossible and are now rejected
    # (treated as None) by parse_existing_csv positivity validation.
    expect("round-trip 0.0 mpix rejected as None",
           parsed_zero[0].spec.mpix, None)
    expect("round-trip 0.0 pitch rejected as None",
           parsed_zero[0].pitch, None)


# --------------------------------------------------------------------------
# deduplicate_specs

def test_deduplicate_specs():
    section("deduplicate_specs")
    import pixelpitch as pp
    from models import Spec

    # Color variants with same specs → unified
    specs = [
        Spec(name="Sony A7 IV schwarz", category="mirrorless", type=None, size=(35.9, 23.9), pitch=5.12, mpix=33.0, year=2021),
        Spec(name="Sony A7 IV silber", category="mirrorless", type=None, size=(35.9, 23.9), pitch=5.12, mpix=33.0, year=2021),
    ]
    deduped = pp.deduplicate_specs(specs)
    expect("color variants unified", len(deduped), 1)
    expect("unified name", deduped[0].name, "Sony A7 IV")

    # Color variants with different specs → kept separate
    specs2 = [
        Spec(name="Cam schwarz", category="fixed", type=None, size=(5.0, 3.7), pitch=2.0, mpix=10.0, year=2020),
        Spec(name="Cam silber", category="fixed", type=None, size=(6.0, 4.5), pitch=3.0, mpix=12.0, year=2020),
    ]
    deduped2 = pp.deduplicate_specs(specs2)
    expect("different specs kept separate", len(deduped2), 2)

    # Parenthetical suffixes stripped (using non-EXTRAS parenthetical)
    specs3 = [
        Spec(name="Canon R5 (Wi-Fi)", category="mirrorless", type=None, size=(36.0, 24.0), pitch=4.39, mpix=45.0, year=2020),
    ]
    deduped3 = pp.deduplicate_specs(specs3)
    expect("parens stripped", deduped3[0].name, "Canon R5")

    # Exact duplicates without EXTRAS → deduplicated
    specs4 = [
        Spec(name="Nikon Z9", category="mirrorless", type=None, size=(35.9, 23.9), pitch=4.35, mpix=45.7, year=2021),
        Spec(name="Nikon Z9", category="mirrorless", type=None, size=(35.9, 23.9), pitch=4.35, mpix=45.7, year=2021),
    ]
    deduped4 = pp.deduplicate_specs(specs4)
    expect("exact duplicates removed", len(deduped4), 1)

    # Empty input
    deduped_empty = pp.deduplicate_specs([])
    expect("empty input", len(deduped_empty), 0)

    # Same specs but different categories -> both kept
    specs5 = [
        Spec(name="Canon R5", category="mirrorless", type=None, size=(36.0, 24.0), pitch=4.39, mpix=45.0, year=2020),
        Spec(name="Canon R5", category="dslr", type=None, size=(36.0, 24.0), pitch=4.39, mpix=45.0, year=2020),
    ]
    deduped5 = pp.deduplicate_specs(specs5)
    expect("different categories kept separate", len(deduped5), 2)

    # Multi-paren name: only last parenthetical should be stripped
    specs6 = [
        Spec(name="Canon EOS (R5) (Kit)", category="mirrorless", type=None, size=(36.0, 24.0), pitch=4.39, mpix=45.0, year=2020),
    ]
    deduped6 = pp.deduplicate_specs(specs6)
    expect("multi-paren: inner parens preserved", deduped6[0].name, "Canon EOS (R5)")

    # EXTRAS word inside a name should NOT match (word boundary test)
    specs7 = [
        Spec(name="Polaroid BodyCam One", category="fixed", type=None, size=(5.0, 3.7), pitch=2.0, mpix=10.0, year=2020),
    ]
    deduped7 = pp.deduplicate_specs(specs7)
    expect("EXTRAS word inside name not matched", deduped7[0].name, "Polaroid BodyCam One")


# --------------------------------------------------------------------------
# merge_camera_data

def test_merge_camera_data():
    section("merge_camera_data")
    import pixelpitch as pp
    from models import Spec

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

    # Duplicate keys within new_specs → deduplicated
    new5 = [
        derive("Canon EOS 250D", "dslr", (22.3, 14.9), 24.1, 2019),
        derive("Canon EOS 250D", "dslr", (22.3, 14.9), 24.1, 2019),
    ]
    merged5 = pp.merge_camera_data(new5, [])
    expect("merge: dedup new_specs same key",
           sum(1 for s in merged5 if s.spec.name == "Canon EOS 250D"), 1)

    # Duplicate keys within new_specs with existing → still deduplicated
    existing6 = [derive("Canon EOS 250D", "dslr", (22.3, 14.9), 24.1, 2019)]
    new6 = [
        derive("Canon EOS 250D", "dslr", (22.3, 14.9), 24.1, 2019),
        derive("Canon EOS 250D", "dslr", (22.3, 14.9), 24.1, 2019),
    ]
    merged6 = pp.merge_camera_data(new6, existing6)
    expect("merge: dedup new_specs with existing",
           sum(1 for s in merged6 if s.spec.name == "Canon EOS 250D"), 1)


def test_merge_field_preservation():
    section("merge_camera_data field preservation")
    import pixelpitch as pp
    from models import Spec, SpecDerived

    def derive(name, category, size, mpix, year, type_val=None, pitch_val=None):
        spec = Spec(name=name, category=category, type=type_val,
                    size=size, pitch=pitch_val, mpix=mpix, year=year)
        return pp.derive_spec(spec)

    # Type preservation: new has None, existing has '1/2.3'
    existing_t = [derive("Cam T", "fixed", (5.0, 3.7), 10.0, 2020, type_val="1/2.3")]
    new_t = [derive("Cam T", "fixed", (5.0, 3.7), 10.0, 2020, type_val=None)]
    merged_t = pp.merge_camera_data(new_t, existing_t)
    expect("merge: preserves type from existing",
           merged_t[0].spec.type, "1/2.3")

    # Size preservation: new has None, existing has size
    existing_s = [derive("Cam S", "fixed", (5.0, 3.7), 10.0, 2020)]
    new_s = [derive("Cam S", "fixed", None, 10.0, 2020)]
    merged_s = pp.merge_camera_data(new_s, existing_s)
    expect("merge: preserves size from existing",
           merged_s[0].spec.size, (5.0, 3.7), tol=0.01)
    # SpecDerived fields must also be preserved (template reads these)
    expect("merge: preserves derived.size from existing",
           merged_s[0].size, (5.0, 3.7), tol=0.01)
    expect("merge: preserves derived.area from existing",
           merged_s[0].area, 18.5, tol=0.01)

    # Pitch preservation: new has None, existing has pitch
    # Need mpix=None so derive_spec doesn't compute pitch from area+mpix
    existing_p = [derive("Cam P", "fixed", (5.0, 3.7), None, 2020, pitch_val=2.0)]
    new_p = [derive("Cam P", "fixed", (5.0, 3.7), None, 2020, pitch_val=None)]
    merged_p = pp.merge_camera_data(new_p, existing_p)
    expect("merge: preserves pitch from existing",
           merged_p[0].spec.pitch, 2.0, tol=0.01)
    expect("merge: preserves derived.pitch from existing",
           merged_p[0].pitch, 2.0, tol=0.01)

    # New values still override existing values
    existing_ov = [derive("Cam OV", "fixed", (5.0, 3.7), 10.0, 2020,
                          type_val="1/2.3", pitch_val=2.0)]
    new_ov = [derive("Cam OV", "fixed", (7.6, 5.7), 12.0, 2021,
                     type_val="1/1.7", pitch_val=3.0)]
    merged_ov = pp.merge_camera_data(new_ov, existing_ov)
    expect("merge: new type overrides existing",
           merged_ov[0].spec.type, "1/1.7")
    expect("merge: new size overrides existing",
           merged_ov[0].spec.size, (7.6, 5.7), tol=0.01)
    expect("merge: new pitch overrides existing",
           merged_ov[0].spec.pitch, 3.0, tol=0.01)

    # mpix preservation: new has None, existing has mpix
    existing_mpx = [derive("Cam M", "fixed", (5.0, 3.7), 10.0, 2020)]
    new_mpx = [derive("Cam M", "fixed", (5.0, 3.7), None, 2020)]
    merged_mpx = pp.merge_camera_data(new_mpx, existing_mpx)
    expect("merge: preserves mpix from existing",
           merged_mpx[0].spec.mpix, 10.0, tol=0.1)

    # spec/derived pitch consistency: new has spec.pitch=None with computed
    # derived.pitch (from area+mpix), existing has spec.pitch=2.0 (direct
    # measurement).  After merge, derived.pitch must equal spec.pitch (the
    # authoritative value), not the computed approximation.
    existing_pc = [derive("Cam PC", "fixed", (5.0, 3.7), 10.0, 2020, pitch_val=2.0)]
    new_pc = [derive("Cam PC", "fixed", (5.0, 3.7), 10.0, 2020, pitch_val=None)]
    merged_pc = pp.merge_camera_data(new_pc, existing_pc)
    expect("merge: spec.pitch preserved from existing",
           merged_pc[0].spec.pitch, 2.0, tol=0.01)
    expect("merge: derived.pitch consistent with spec.pitch",
           merged_pc[0].pitch, 2.0, tol=0.01)

    # matched_sensors preservation: new has None (sensors_db unavailable),
    # existing has ['IMX455'] — must preserve from existing.
    existing_ms = SpecDerived(
        spec=Spec(name='Cam MS', category='dslr', type=None,
                  size=(36.0, 24.0), pitch=None, mpix=45.0, year=2020),
        size=(36.0, 24.0), area=864.0, pitch=4.39,
        matched_sensors=['IMX455'], id=0
    )
    # derive_spec with empty sensors_db returns matched_sensors=None
    new_ms = pp.derive_spec(
        Spec(name='Cam MS', category='dslr', type=None,
             size=(36.0, 24.0), pitch=None, mpix=45.0, year=2020),
        sensors_db={}
    )
    merged_ms = pp.merge_camera_data([new_ms], [existing_ms])
    expect("merge: preserves matched_sensors from existing when new is None",
           merged_ms[0].matched_sensors, ['IMX455'])

    # matched_sensors=[] is authoritative (checked, found nothing) — should NOT
    # be overridden by existing data.  Create a sensors_db with no match.
    existing_ms2 = SpecDerived(
        spec=Spec(name='Cam MS2', category='dslr', type=None,
                  size=(36.0, 24.0), pitch=None, mpix=45.0, year=2020),
        size=(36.0, 24.0), area=864.0, pitch=4.39,
        matched_sensors=['IMX455'], id=0
    )
    new_ms2 = pp.derive_spec(
        Spec(name='Cam MS2', category='dslr', type=None,
             size=(36.0, 24.0), pitch=None, mpix=45.0, year=2020),
        sensors_db={'NOMATCH': {'sensor_width_mm': 99.0, 'sensor_height_mm': 99.0, 'megapixels': [99.0]}}
    )
    merged_ms2 = pp.merge_camera_data([new_ms2], [existing_ms2])
    expect("merge: [] from checked db does not preserve existing",
           merged_ms2[0].matched_sensors, [])


# --------------------------------------------------------------------------
# merge_camera_data year-change log

def test_merge_size_consistency():
    """Verify merge_camera_data keeps spec.size and derived.size consistent
    when spec.size is preserved from existing but derived.size was type-computed."""
    section("merge size consistency")
    import pixelpitch as pp
    from models import Spec

    # Existing: measured size from Geizhals
    existing_spec = Spec(name='Test Cam', category='fixed', type='1/2.3',
                         size=(5.76, 4.29), pitch=None, mpix=12.0, year=2020)
    existing = pp.derive_spec(existing_spec)
    existing.id = 0

    # New: type-derived size from source (no explicit size)
    new_spec = Spec(name='Test Cam', category='fixed', type='1/2.3',
                    size=None, pitch=None, mpix=12.0, year=2020)
    new = pp.derive_spec(new_spec)

    merged = pp.merge_camera_data([new], [existing])
    m = merged[0]

    # spec.size should be preserved from existing (measured value)
    expect("merge size consistency: spec.size preserved",
           m.spec.size, (5.76, 4.29), tol=0.01)
    # derived.size must match spec.size (not the type-computed value)
    expect("merge size consistency: derived.size matches spec.size",
           m.size, m.spec.size, tol=0.01)
    # derived.area must be consistent with derived.size
    expect("merge size consistency: area consistent",
           abs(m.area - m.size[0] * m.size[1]) < 0.01, True)
    # derived.pitch must be consistent with derived.area (or from spec.pitch)
    if m.spec.pitch is None:
        expected_pitch = pp.pixel_pitch(m.size[0] * m.size[1], m.spec.mpix)
        if expected_pitch == 0.0:
            expected_pitch = None
        expect("merge size consistency: pitch consistent with correct area",
               abs(m.pitch - expected_pitch) < 0.01 if m.pitch and expected_pitch else m.pitch == expected_pitch, True)

    # Also test case where TYPE_SIZE matches Geizhals (no inconsistency expected)
    existing_spec2 = Spec(name='Match Cam', category='fixed', type='1/2.3',
                          size=(6.17, 4.55), pitch=None, mpix=12.0, year=2020)
    existing2 = pp.derive_spec(existing_spec2)
    existing2.id = 0

    new_spec2 = Spec(name='Match Cam', category='fixed', type='1/2.3',
                     size=None, pitch=None, mpix=12.0, year=2020)
    new2 = pp.derive_spec(new_spec2)

    merged2 = pp.merge_camera_data([new2], [existing2])
    m2 = merged2[0]
    expect("merge size consistency: matching sizes stay consistent",
           m2.size, m2.spec.size, tol=0.01)


def test_merge_gsmarena_measured_preserved():
    """Verify merge_camera_data preserves measured Geizhals spec.size
    when GSMArena provides only spec.type (spec.size=None)."""
    section("merge GSMArena measured preservation")
    import pixelpitch as pp
    from models import Spec

    # Existing: measured size from Geizhals (slightly different from TYPE_SIZE)
    existing_spec = Spec(name='Phone X', category='smartphone', type='1/1.3',
                         size=(9.76, 7.30), pitch=None, mpix=200.0, year=2025)
    existing = pp.derive_spec(existing_spec)
    existing.id = 0

    # New: spec.size=None, spec.type='1/1.3' (GSMArena after provenance fix)
    new_spec = Spec(name='Phone X', category='smartphone', type='1/1.3',
                    size=None, pitch=None, mpix=200.0, year=2025)
    new = pp.derive_spec(new_spec)

    merged = pp.merge_camera_data([new], [existing])
    m = merged[0]

    # spec.size should be preserved from existing (measured Geizhals value)
    expect("merge GSMArena: spec.size preserved from measured",
           m.spec.size, (9.76, 7.30), tol=0.01)
    # derived.size must match spec.size (not TYPE_SIZE)
    expect("merge GSMArena: derived.size matches spec.size",
           m.size, m.spec.size, tol=0.01)
    # derived.area must be consistent with derived.size
    expect("merge GSMArena: area consistent",
           abs(m.area - m.size[0] * m.size[1]) < 0.01, True)

    # Also test case where GSMArena provides spec.size=None but there is no
    # existing data — derived.size should come from type lookup
    new_only_spec = Spec(name='New Phone', category='smartphone', type='1/1.3',
                          size=None, pitch=None, mpix=200.0, year=2025)
    new_only = pp.derive_spec(new_only_spec)
    merged_new = pp.merge_camera_data([new_only], [])
    expect("merge GSMArena no existing: derived.size from type",
           merged_new[0].size, (9.84, 7.40), tol=0.01)


def test_merge_year_change_log():
    section("merge_camera_data year-change log")
    import contextlib
    import pixelpitch as pp
    from models import Spec

    def derive(name, category, size, mpix, year, pitch_val=None):
        spec = Spec(name=name, category=category, type=None,
                    size=size, pitch=pitch_val, mpix=mpix, year=year)
        return pp.derive_spec(spec)

    # Case 1: years differ AND pitch is preserved from existing
    # (This was the case where the elif was unreachable before the fix)
    # Use mpix=None for new so that derive_spec produces derived.pitch=None
    # (if mpix is set, pitch gets computed from area+mpix and is not None).
    existing = [derive("Cam YP", "fixed", (5.0, 3.7), 10.0, 2020, pitch_val=2.0)]
    new = [derive("Cam YP", "fixed", (5.0, 3.7), None, 2021, pitch_val=None)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        merged = pp.merge_camera_data(new, existing)
    output = buf.getvalue()
    expect("year-change log with pitch preservation",
           "Year changed" in output, True)
    expect("year correctly updated", merged[0].spec.year, 2021)
    expect("pitch preserved from existing", merged[0].pitch, 2.0, tol=0.01)

    # Case 2: years differ, no pitch preservation needed
    existing2 = [derive("Cam Y2", "fixed", (5.0, 3.7), 10.0, 2020)]
    new2 = [derive("Cam Y2", "fixed", (5.0, 3.7), 10.0, 2021)]
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        merged2 = pp.merge_camera_data(new2, existing2)
    output2 = buf2.getvalue()
    expect("year-change log without pitch preservation",
           "Year changed" in output2, True)


# --------------------------------------------------------------------------
# Sony DSC hyphen normalisation

def test_sony_dsc_hyphen_normalisation():
    section("Sony DSC hyphen normalisation")
    from sources import imaging_resource

    # Model Name path: "Sony DSC-HX400" should normalise to "Sony DSC HX400"
    name_model = imaging_resource._parse_camera_name(
        {"Model Name": "Sony DSC-HX400"},
        "https://www.imaging-resource.com/cameras/sony-dsc-hx400-review/specifications/"
    )
    expect("DSC-HX400 from Model Name", name_model, "Sony DSC HX400")

    # Model Name path: "Sony DSC-WX350" should normalise to "Sony DSC WX350"
    name_model2 = imaging_resource._parse_camera_name(
        {"Model Name": "Sony DSC-WX350"},
        "https://www.imaging-resource.com/cameras/sony-dsc-wx350-review/specifications/"
    )
    expect("DSC-WX350 from Model Name", name_model2, "Sony DSC WX350")

    # URL fallback path (already works via .replace("-", " "))
    name_url = imaging_resource._parse_camera_name(
        {"Model Name": ""},
        "https://www.imaging-resource.com/cameras/sony-dsc-hx400-review/specifications/"
    )
    expect("DSC-HX400 from URL fallback", name_url, "Sony DSC HX400")


# --------------------------------------------------------------------------
# sensor_size_from_type

def test_sensor_size_from_type():
    section("sensor_size_from_type")
    import pixelpitch as pp

    # Type in lookup table → returns table value
    result = pp.sensor_size_from_type("1/2.3")
    expect("1/2.3 from table", result, (6.17, 4.55), tol=0.01)

    # Type not in lookup table but starts with "1/" → computed
    result3 = pp.sensor_size_from_type("1/3.1")
    expect("1/3.1 computed (not in table)", result3 is not None, True)
    # Computed value will be approximate
    if result3:
        expect("1/3.1 width > 0", result3[0] > 0, True)
        expect("1/3.1 height > 0", result3[1] > 0, True)

    # None type
    result4 = pp.sensor_size_from_type(None)
    expect("None type returns None", result4, None)

    # Unknown type (not 1/x format, not in table)
    result5 = pp.sensor_size_from_type("APS-C")
    expect("unknown type returns None", result5, None)

    # Phone-format sensor types (merged from gsmarena.PHONE_TYPE_SIZE)
    result6 = pp.sensor_size_from_type("1/1.3")
    expect("1/1.3 measured width", result6[0], 9.84, tol=0.01)
    expect("1/1.3 measured height", result6[1], 7.40, tol=0.01)

    result7 = pp.sensor_size_from_type("1/1.7")
    expect("1/1.7 measured width", result7[0], 7.60, tol=0.01)
    expect("1/1.7 measured height", result7[1], 5.70, tol=0.01)

    result8 = pp.sensor_size_from_type("1/2.8")
    expect("1/2.8 measured width", result8[0], 5.12, tol=0.01)
    expect("1/2.8 measured height", result8[1], 3.84, tol=0.01)

    # Invalid fractional types — must return None, not crash
    result9 = pp.sensor_size_from_type("1/0")
    expect("1/0 returns None (ZeroDivisionError guard)", result9, None)

    result10 = pp.sensor_size_from_type("1/0.0")
    expect("1/0.0 returns None (ZeroDivisionError guard)", result10, None)

    result11 = pp.sensor_size_from_type("1/")
    expect("1/ returns None (ValueError guard)", result11, None)

    result12 = pp.sensor_size_from_type("1/-1")
    expect("1/-1 returns None (negative diagonal)", result12, None)


def test_parse_sensor_field():
    section("parse_sensor_field")
    import pixelpitch as pp

    # Fractional-inch sensor type
    result1 = pp.parse_sensor_field('CMOS 1/2.3"')
    expect("fractional type 1/2.3", result1["type"], "1/2.3")

    # Bare 1-inch sensor type (not fractional 1/x.y)
    result2 = pp.parse_sensor_field('CMOS 1"')
    expect("bare 1-inch type", result2["type"], "1")

    # 1-inch with -inch suffix
    result3 = pp.parse_sensor_field('CMOS 1-inch')
    expect("1-inch suffix type", result3["type"], "1")

    # 1-inch with " inch" (space before inch)
    result4 = pp.parse_sensor_field('CMOS 1 inch')
    expect("1 inch suffix type", result4["type"], "1")

    # Fractional type takes precedence over bare 1-inch
    result5 = pp.parse_sensor_field('CMOS 1/1.7"')
    expect("fractional takes precedence", result5["type"], "1/1.7")

    # Empty input
    result6 = pp.parse_sensor_field('')
    expect("empty input type", result6["type"], None)

    # With mm dimensions and type
    result7 = pp.parse_sensor_field('CMOS 1", 13.2x8.8mm')
    expect("1-inch with mm dims type", result7["type"], "1")
    expect("1-inch with mm dims size", result7["size"], (13.2, 8.8), tol=0.01)

    # SIZE_MM_RE handles Unicode multiplication sign (U+00D7)
    result8 = pp.parse_sensor_field('CMOS 36.0×24.0mm')
    expect("SIZE handles Unicode ×", result8["size"], (36.0, 24.0), tol=0.01)

    # SIZE_MM_RE handles spaces around x
    result9 = pp.parse_sensor_field('CMOS 36.0 x 24.0 mm')
    expect("SIZE handles spaces around x", result9["size"], (36.0, 24.0), tol=0.01)

    # PITCH_UM_RE handles Greek mu (U+03BC)
    result10 = pp.parse_sensor_field('CMOS 5.12μm')
    expect("PITCH handles Greek mu", result10["pitch"], 5.12, tol=0.01)

    # PITCH_UM_RE handles "microns" suffix
    result11 = pp.parse_sensor_field('CMOS 5.12 microns')
    expect("PITCH handles microns suffix", result11["pitch"], 5.12, tol=0.01)

    # PITCH_UM_RE handles lowercase ASCII "um"
    result_um = pp.parse_sensor_field('CMOS 5.12um')
    expect("PITCH handles um", result_um["pitch"], 5.12, tol=0.01)

    # ValueError guard: malformed float in dimension string
    result12 = pp.parse_sensor_field('CMOS 36.0.1x24.0mm')
    expect("malformed size returns None", result12["size"], None)

    # ValueError guard: malformed float in pitch string
    result13 = pp.parse_sensor_field('CMOS 5.1.2µm')
    expect("malformed pitch returns None", result13["pitch"], None)


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

    # Edge case: zero mpix — must return 0.0, not crash
    pitch4 = pp.pixel_pitch(864.0, 0.0)
    expect("zero mpix pitch", pitch4, 0.0)

    # Edge case: negative mpix — must return 0.0, not crash
    pitch5 = pp.pixel_pitch(864.0, -1.0)
    expect("negative mpix pitch", pitch5, 0.0)

    # Edge case: negative area — must return 0.0, not crash (ValueError guard)
    pitch6 = pp.pixel_pitch(-864.0, 33.0)
    expect("negative area pitch", pitch6, 0.0)

    # Edge case: zero area — must return 0.0, not crash
    pitch7 = pp.pixel_pitch(0.0, 33.0)
    expect("zero area pitch", pitch7, 0.0)

    # Edge case: NaN area — must return 0.0, not propagate NaN
    pitch8 = pp.pixel_pitch(float('nan'), 33.0)
    expect("nan area pitch", pitch8, 0.0)

    # Edge case: NaN mpix — must return 0.0, not propagate NaN
    pitch9 = pp.pixel_pitch(864.0, float('nan'))
    expect("nan mpix pitch", pitch9, 0.0)

    # Edge case: inf area — must return 0.0, not propagate inf
    pitch10 = pp.pixel_pitch(float('inf'), 33.0)
    expect("inf area pitch", pitch10, 0.0)

    # Edge case: inf mpix — must return 0.0, not propagate inf
    pitch11 = pp.pixel_pitch(864.0, float('inf'))
    expect("inf mpix pitch", pitch11, 0.0)


def test_derive_spec_zero_pitch():
    """Verify derive_spec treats invalid direct spec.pitch values (0.0, negative, NaN) as None.

    Direct spec.pitch values that are non-positive or non-finite are treated as
    invalid. derive_spec falls through to the computed path when possible, or
    produces None when no valid pitch can be determined.
    """
    section("derive_spec invalid direct pitch handling")
    import pixelpitch as pp
    from models import Spec

    # spec.pitch=0.0 with size and mpix — invalid direct pitch, fall through
    # to computed path (pixel_pitch(area, mpix) ≈ 5.12)
    spec_zero = Spec(name="Zero Pitch Cam", category="fixed", type=None,
                     size=(35.9, 23.9), pitch=0.0, mpix=33.0, year=2021)
    d_zero = pp.derive_spec(spec_zero)
    expect("derive_spec: spec.pitch=0.0 falls through to computed",
           d_zero.pitch is not None, True)
    expect("derive_spec: fallback pitch ≈ 5.12",
           d_zero.pitch, 5.12, tol=0.05)

    # spec.pitch=0.0 with mpix=0.0 — invalid direct, computed also invalid → None
    spec_zero_nocalc = Spec(name="Zero Pitch No Calc", category="fixed", type=None,
                             size=(5.0, 3.7), pitch=0.0, mpix=0.0, year=2020)
    d_zero_nocalc = pp.derive_spec(spec_zero_nocalc)
    expect("derive_spec: spec.pitch=0.0 with mpix=0.0 → None",
           d_zero_nocalc.pitch, None)

    # spec.pitch=None with size and mpix — derive_spec must compute
    spec_none = Spec(name="None Pitch Cam", category="fixed", type=None,
                     size=(35.9, 23.9), pitch=None, mpix=33.0, year=2021)
    d_none = pp.derive_spec(spec_none)
    expect("derive_spec: spec.pitch=None computes from area+mpix",
           d_none.pitch is not None, True)
    expect("derive_spec: computed pitch ≈ 5.12",
           d_none.pitch, 5.12, tol=0.05)

    # spec.pitch=-1.0 with size and mpix — invalid direct, fall through to computed
    spec_neg = Spec(name="Neg Pitch Cam", category="fixed", type=None,
                    size=(35.9, 23.9), pitch=-1.0, mpix=33.0, year=2021)
    d_neg = pp.derive_spec(spec_neg)
    expect("derive_spec: spec.pitch=-1.0 falls through to computed",
           d_neg.pitch is not None, True)
    expect("derive_spec: negative pitch fallback ≈ 5.12",
           d_neg.pitch, 5.12, tol=0.05)

    # spec.pitch=nan with size and mpix — invalid direct, fall through to computed
    spec_nan = Spec(name="NaN Pitch Cam", category="fixed", type=None,
                    size=(35.9, 23.9), pitch=float('nan'), mpix=33.0, year=2021)
    d_nan = pp.derive_spec(spec_nan)
    expect("derive_spec: spec.pitch=nan falls through to computed",
           d_nan.pitch is not None, True)
    expect("derive_spec: nan pitch fallback ≈ 5.12",
           d_nan.pitch, 5.12, tol=0.05)

    # spec.pitch=-1.0 with no mpix — invalid direct, no computed fallback → None
    spec_neg_nocalc = Spec(name="Neg Pitch No Calc", category="fixed", type=None,
                           size=(5.0, 3.7), pitch=-1.0, mpix=None, year=2020)
    d_neg_nocalc = pp.derive_spec(spec_neg_nocalc)
    expect("derive_spec: spec.pitch=-1.0 with no mpix → None",
           d_neg_nocalc.pitch, None)


def test_derive_spec_negative_size():
    """Verify derive_spec handles invalid sensor dimensions gracefully.

    Negative, zero, NaN, and inf dimensions are physically meaningless.
    derive_spec should set size=None and area=None (not propagate NaN/inf
    or produce a misleading 0.0 sentinel pitch).
    """
    section("derive_spec invalid size handling")
    import pixelpitch as pp
    from models import Spec

    # Negative width: size=(-5.0, 3.7), pitch=None, mpix=10.0
    spec_neg = Spec(name="Neg Size Cam", category="fixed", type=None,
                     size=(-5.0, 3.7), pitch=None, mpix=10.0, year=2020)
    d_neg = pp.derive_spec(spec_neg)
    expect("derive_spec negative size: no crash", d_neg is not None, True)
    expect("derive_spec negative size: size is None", d_neg.size, None)
    expect("derive_spec negative size: area is None", d_neg.area, None)
    expect("derive_spec negative size: pitch is None", d_neg.pitch, None)

    # Negative height: size=(5.0, -3.7), pitch=None, mpix=10.0
    spec_neg2 = Spec(name="Neg Height Cam", category="fixed", type=None,
                      size=(5.0, -3.7), pitch=None, mpix=10.0, year=2020)
    d_neg2 = pp.derive_spec(spec_neg2)
    expect("derive_spec negative height: no crash", d_neg2 is not None, True)
    expect("derive_spec negative height: size is None", d_neg2.size, None)
    expect("derive_spec negative height: area is None", d_neg2.area, None)
    expect("derive_spec negative height: pitch is None", d_neg2.pitch, None)

    # NaN width: size=(nan, 24.0), pitch=None, mpix=10.0
    spec_nan = Spec(name="NaN Size Cam", category="fixed", type=None,
                     size=(float('nan'), 24.0), pitch=None, mpix=10.0, year=2020)
    d_nan = pp.derive_spec(spec_nan)
    expect("derive_spec NaN size: no crash", d_nan is not None, True)
    expect("derive_spec NaN size: size is None", d_nan.size, None)
    expect("derive_spec NaN size: area is None", d_nan.area, None)
    expect("derive_spec NaN size: pitch is None", d_nan.pitch, None)

    # inf width: size=(inf, 24.0), pitch=None, mpix=10.0
    spec_inf = Spec(name="Inf Size Cam", category="fixed", type=None,
                     size=(float('inf'), 24.0), pitch=None, mpix=10.0, year=2020)
    d_inf = pp.derive_spec(spec_inf)
    expect("derive_spec inf size: no crash", d_inf is not None, True)
    expect("derive_spec inf size: size is None", d_inf.size, None)
    expect("derive_spec inf size: area is None", d_inf.area, None)
    expect("derive_spec inf size: pitch is None", d_inf.pitch, None)

    # Zero width: size=(0.0, 24.0), pitch=None, mpix=10.0
    spec_zero = Spec(name="Zero Size Cam", category="fixed", type=None,
                      size=(0.0, 24.0), pitch=None, mpix=10.0, year=2020)
    d_zero = pp.derive_spec(spec_zero)
    expect("derive_spec zero size: no crash", d_zero is not None, True)
    expect("derive_spec zero size: size is None", d_zero.size, None)
    expect("derive_spec zero size: area is None", d_zero.area, None)
    expect("derive_spec zero size: pitch is None", d_zero.pitch, None)


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

    # megapixels=0.0 → must not crash (ZeroDivisionError guard)
    matches5 = pp.match_sensors(36.0, 24.0, 0.0, sensors_db)
    expect("match with zero mpix: no crash", isinstance(matches5, list), True)
    expect("match with zero mpix: size-only match",
           "IMX455" in matches5, True)

    # width=0.0 → must return [] (not crash)
    matches6 = pp.match_sensors(0.0, 24.0, 61.0, sensors_db)
    expect("match with zero width", matches6, [])

    # height=0.0 → must return [] (not crash)
    matches7 = pp.match_sensors(36.0, 0.0, 61.0, sensors_db)
    expect("match with zero height", matches7, [])


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


def test_load_csv():
    section("load_csv error handling")
    import tempfile
    import pixelpitch as pp
    from pathlib import Path

    # UnicodeDecodeError should return None gracefully
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Write a file with invalid UTF-8 bytes
        bad_file = tmpdir_path / "camera-data.csv"
        bad_file.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")
        result = pp.load_csv(tmpdir_path)
    expect("UnicodeDecodeError returns None", result, None)

    # OSError should return None gracefully
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Use a directory as the path (causes IsADirectoryError, a subclass of OSError)
        bad_dir = tmpdir_path / "camera-data.csv"
        bad_dir.mkdir()
        result2 = pp.load_csv(tmpdir_path)
    expect("OSError returns None", result2, None)

    # Non-existent file returns None
    with tempfile.TemporaryDirectory() as tmpdir:
        result3 = pp.load_csv(Path(tmpdir))
    expect("missing file returns None", result3, None)

    # Valid file returns content
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        good_file = tmpdir_path / "camera-data.csv"
        good_file.write_text("id,name\n0,Test\n", encoding="utf-8")
        result4 = pp.load_csv(tmpdir_path)
    expect("valid file returns content", result4 is not None, True)
    expect("valid file content starts with header", result4.startswith("id,name") if result4 else False, True)


# --------------------------------------------------------------------------
# about.html template rendering

def test_about_html_rendering():
    section("about.html rendering")
    import pixelpitch as pp
    from datetime import datetime, timezone

    date = datetime.now(timezone.utc)
    html = pp._get_env().get_template("about.html").render(page="about", date=date)

    expect("title contains About Pixel Pitch",
           "About Pixel Pitch" in html, True)
    expect("LD+JSON has AboutPage type",
           '"@type": "AboutPage"' in html, True)
    expect("top-level @type is AboutPage (not Dataset)",
           html.index('"@type": "AboutPage"') < html.index('"@type": "Dataset"'), True)


# --------------------------------------------------------------------------
# GSMArena _select_main_lens edge cases

def test_gsmarena_select_main_lens():
    section("GSMArena _select_main_lens edge cases")

    # Two wide lenses — first wide should be selected (stable sort preserves order)
    cam_two_wide = "12 MP, f/2.2, (wide)\n50 MP, f/1.7, (wide)"
    result1 = gsmarena._select_main_lens(cam_two_wide)
    expect("two wide lenses: returns a result", result1 is not None, True)
    if result1:
        expect("two wide lenses: picks first wide",
               result1.startswith("12 MP"), True)

    # No wide lenses — telephoto + ultrawide
    cam_no_wide = "10 MP, f/2.4, (telephoto)\n8 MP, f/2.2, (ultrawide)"
    result2 = gsmarena._select_main_lens(cam_no_wide)
    expect("no wide lenses: returns a result", result2 is not None, True)
    if result2:
        expect("no wide lenses: picks ultrawide over telephoto",
               "ultrawide" in result2.lower() or "ultra wide" in result2.lower(), True)

    # Empty camera value
    result3 = gsmarena._select_main_lens("")
    expect("empty camera value", result3 is None, True)

    # Lens with no role tag (priority 1, between wide=0 and ultrawide=3)
    cam_no_role = "48 MP, f/1.8\n8 MP, f/2.4, (ultrawide)"
    result4 = gsmarena._select_main_lens(cam_no_role)
    expect("no role tag: returns a result", result4 is not None, True)
    if result4:
        expect("no role tag: picks untagged over ultrawide",
               result4.startswith("48 MP"), True)

    # Decimal MP values — regex split must not break at the decimal point
    # (regression test for C45-01: \b word boundary broke "12.2 MP" into
    # "12." and "2 MP", causing mpix=2.0 instead of 12.2)
    cam_decimal = '12.2 MP, f/1.9, (wide), 1/2.55", 1.25µm'
    result5 = gsmarena._select_main_lens(cam_decimal)
    expect("decimal MP: returns a result", result5 is not None, True)
    if result5:
        expect("decimal MP: starts with full '12.2 MP'",
               result5.startswith("12.2 MP"), True)

    # Decimal MP periscope lens
    cam_periscope = "10.7 MP, f/4.3, 240mm (periscope)"
    result6 = gsmarena._select_main_lens(cam_periscope)
    expect("decimal periscope: returns a result", result6 is not None, True)
    if result6:
        expect("decimal periscope: starts with full '10.7 MP'",
               result6.startswith("10.7 MP"), True)

    # Decimal MP depth camera
    cam_depth = "0.3 MP, f/2.4, (depth)"
    result7 = gsmarena._select_main_lens(cam_depth)
    expect("decimal depth: returns a result", result7 is not None, True)
    if result7:
        expect("decimal depth: starts with full '0.3 MP'",
               result7.startswith("0.3 MP"), True)

    # Multi-lens with mixed integer and decimal MP
    cam_mixed = '50 MP, f/1.7, (wide), 1/1.3", 0.6µm\n12.2 MP, f/2.2, (ultrawide), 1/2.55", 1.4µm'
    result8 = gsmarena._select_main_lens(cam_mixed)
    expect("mixed int/decimal: returns a result", result8 is not None, True)
    if result8:
        expect("mixed int/decimal: picks 50 MP wide over 12.2 MP ultrawide",
               result8.startswith("50 MP"), True)


def test_gsmarena_decimal_mp():
    """Verify _phone_to_spec correctly handles decimal megapixel camera values.

    Regression test for C45-01: the regex split in _select_main_lens broke
    decimal MP values, causing wrong mpix (12.2→2.0) and lost sensor type.
    """
    section("GSMArena decimal MP in _phone_to_spec")

    # Decimal MP main camera with full sensor info
    fields = {
        "Main Camera": '12.2 MP, f/1.9, 25mm (wide), 1/2.55", 1.25µm, dual pixel PDAF, OIS'
    }
    spec = gsmarena._phone_to_spec("Google Pixel 7", fields)
    expect("decimal MP: spec is not None", spec is not None, True)
    if spec:
        expect("decimal MP: mpix=12.2", spec.mpix, 12.2, tol=0.1)
        expect("decimal MP: type=1/2.55", spec.type, "1/2.55")
        expect("decimal MP: pitch=1.25", spec.pitch, 1.25, tol=0.01)
        expect("decimal MP: category", spec.category, "smartphone")
        expect("decimal MP: spec.size is None (type-derived)", spec.size, None)

    # Decimal MP with integer secondary lenses
    fields2 = {
        "Quad": '50 MP, f/1.7, (wide), 1/1.3", 0.6µm\n10.7 MP, f/4.3, 240mm (periscope), 1/3.52", 1.12µm\n12 MP, f/2.2, (ultrawide), 1/2.55", 1.4µm'
    }
    spec2 = gsmarena._phone_to_spec("Test Phone", fields2)
    expect("mixed MP: spec is not None", spec2 is not None, True)
    if spec2:
        expect("mixed MP: picks 50 MP wide", spec2.mpix, 50.0, tol=0.1)
        expect("mixed MP: type=1/1.3", spec2.type, "1/1.3")


# --------------------------------------------------------------------------
# create_camera_key — year mismatch across sources must not produce duplicates

def test_create_camera_key_year_mismatch():
    section("create_camera_key year mismatch")
    import pixelpitch as pp
    from models import Spec

    def derive(name, category, size, mpix, year):
        spec = Spec(name=name, category=category, type=None,
                    size=size, pitch=None, mpix=mpix, year=year)
        return pp.derive_spec(spec)

    # Same camera from Geizhals (year=2005) and openMVG (year=None)
    # must merge into a single entry, not produce duplicates.
    existing = [derive("Canon EOS 5D", "dslr", (36.0, 24.0), 12.7, 2005)]
    new = [derive("Canon EOS 5D", "dslr", (36.0, 24.0), 12.7, None)]
    merged = pp.merge_camera_data(new, existing)
    expect("year mismatch: no duplicate",
           sum(1 for s in merged if s.spec.name == "Canon EOS 5D"), 1)

    # Verify existing year is preserved when new has None
    a7iv = [s for s in merged if s.spec.name == "Canon EOS 5D"][0]
    expect("year mismatch: existing year preserved", a7iv.spec.year, 2005)

    # Different years on same camera (new overwrites)
    existing2 = [derive("Sony A7 IV", "mirrorless", (35.9, 23.9), 33.0, 2021)]
    new2 = [derive("Sony A7 IV", "mirrorless", (35.9, 23.9), 33.0, 2022)]
    merged2 = pp.merge_camera_data(new2, existing2)
    expect("different years: no duplicate",
           sum(1 for s in merged2 if s.spec.name == "Sony A7 IV"), 1)


# --------------------------------------------------------------------------
# Category dedup: same camera in multiple Geizhals categories must not
# produce duplicates in the merge result

def test_category_dedup():
    section("category dedup across Geizhals categories")
    import pixelpitch as pp
    from models import Spec

    def derive(name, category, size, mpix, year):
        spec = Spec(name=name, category=category, type=None,
                    size=size, pitch=None, mpix=mpix, year=year)
        return pp.derive_spec(spec)

    # Simulate Geizhals fetching the same camera in two categories
    # (e.g., Canon EOS 5D as both "dslr" and "rangefinder")
    dslr_specs = [derive("Canon EOS 5D", "dslr", (36.0, 24.0), 12.7, 2005)]
    rf_specs = [derive("Canon EOS 5D", "rangefinder", (36.0, 24.0), 12.7, 2005)]
    # When merged, the two entries have different merge keys → both preserved
    all_specs = dslr_specs + rf_specs
    names = [s.spec.name for s in all_specs]
    expect("same camera different categories: 2 raw entries",
           sum(1 for n in names if n == "Canon EOS 5D"), 2)

    # The dedup logic in render_html filters the rangefinder category
    # to remove entries whose name also exists in other categories.
    # Simulate that logic here:
    rf_names = {s.spec.name for s in rf_specs}
    other_names = set()
    for spec in dslr_specs:
        other_names.add(spec.spec.name)
    dup_rf_names = rf_names & other_names
    filtered_rf = [s for s in rf_specs if s.spec.name not in dup_rf_names]
    expect("rangefinder dedup: filtered to 0", len(filtered_rf), 0)

    # Actual rangefinder (Leica M11) is kept because it's not in other categories
    actual_rf = [derive("Leica M11", "rangefinder", (36.0, 24.0), 60.3, 2022)]
    rf_names2 = {s.spec.name for s in actual_rf}
    dup_rf_names2 = rf_names2 & other_names
    filtered_rf2 = [s for s in actual_rf if s.spec.name not in dup_rf_names2]
    expect("actual rangefinder kept", len(filtered_rf2), 1)


# --------------------------------------------------------------------------
# derive_spec: computed pitch from pixel_pitch sentinel

def test_derive_spec_computed_zero_pitch():
    """Verify derive_spec converts pixel_pitch 0.0 sentinel to None.

    pixel_pitch() returns 0.0 for invalid inputs (negative, zero, NaN, inf).
    derive_spec must convert this 0.0 sentinel to None so that:
    - selectattr('pitch', 'ne', None) routes the camera to the correct section
    - write_csv produces empty string instead of "0.00" (consistent round-trip)
    """
    section("derive_spec computed 0.0 pitch -> None")
    import pixelpitch as pp
    from models import Spec

    # Case 1: spec.pitch=None, mpix=0.0 -> pixel_pitch(area, 0.0) = 0.0 -> None
    spec_mpix_zero = Spec(name="Zero MP Cam", category="fixed", type=None,
                          size=(5.0, 3.7), pitch=None, mpix=0.0, year=2020)
    d_mpix_zero = pp.derive_spec(spec_mpix_zero)
    expect("derive_spec: mpix=0.0, pitch=None -> computed pitch is None",
           d_mpix_zero.pitch, None)

    # Case 2: spec.pitch=None, mpix=-1.0 -> pixel_pitch(area, -1.0) = 0.0 -> None
    spec_mpix_neg = Spec(name="Neg MP Cam", category="fixed", type=None,
                         size=(5.0, 3.7), pitch=None, mpix=-1.0, year=2020)
    d_mpix_neg = pp.derive_spec(spec_mpix_neg)
    expect("derive_spec: mpix=-1.0, pitch=None -> computed pitch is None",
           d_mpix_neg.pitch, None)

    # Case 3: spec.pitch=0.0 (direct) is treated as invalid, falls through
    # to computed path (pixel_pitch(area, mpix) ≈ 5.12)
    spec_direct_zero = Spec(name="Direct Zero Cam", category="fixed", type=None,
                            size=(35.9, 23.9), pitch=0.0, mpix=33.0, year=2021)
    d_direct_zero = pp.derive_spec(spec_direct_zero)
    expect("derive_spec: spec.pitch=0.0 direct falls through to computed",
           d_direct_zero.pitch is not None, True)
    expect("derive_spec: spec.pitch=0.0 direct fallback ≈ 5.12",
           d_direct_zero.pitch, 5.12, tol=0.05)

    # Case 4: CSV round-trip for mpix=0.0 -> pitch=None -> no data loss
    import tempfile
    d_mpix_zero.id = 0
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        out_path = Path(f.name)
    try:
        pp.write_csv([d_mpix_zero], out_path)
        csv_text = out_path.read_text(encoding="utf-8")
    finally:
        out_path.unlink(missing_ok=True)
    parsed_back = pp.parse_existing_csv(csv_text)
    expect("round-trip mpix=0.0: pitch stays None",
           parsed_back[0].pitch, None)


# --------------------------------------------------------------------------
# write_csv: non-finite float guards

def test_write_csv_nonfinite_guards():
    """Verify write_csv does not output inf/nan strings for float fields."""
    section("write_csv non-finite float guards")
    import tempfile
    import pixelpitch as pp
    from models import Spec

    # spec.mpix=inf -> CSV should have empty mpix cell
    spec_inf = Spec(name="Inf MP Cam", category="fixed", type=None,
                    size=(35.9, 23.9), pitch=5.12, mpix=float('inf'), year=2020)
    d_inf = pp.derive_spec(spec_inf)
    d_inf.id = 0
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        out_path = Path(f.name)
    try:
        pp.write_csv([d_inf], out_path)
        csv_text = out_path.read_text(encoding="utf-8")
    finally:
        out_path.unlink(missing_ok=True)
    row = csv_text.splitlines()[1]
    expect("write_csv inf mpix: no 'inf' in CSV row", "inf" not in row, True)
    expect("write_csv inf mpix: no 'nan' in CSV row", "nan" not in row, True)

    # spec.mpix=nan -> CSV should have empty mpix cell
    spec_nan = Spec(name="NaN MP Cam", category="fixed", type=None,
                    size=(35.9, 23.9), pitch=5.12, mpix=float('nan'), year=2020)
    d_nan = pp.derive_spec(spec_nan)
    d_nan.id = 0
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        out_path2 = Path(f.name)
    try:
        pp.write_csv([d_nan], out_path2)
        csv_text2 = out_path2.read_text(encoding="utf-8")
    finally:
        out_path2.unlink(missing_ok=True)
    row2 = csv_text2.splitlines()[1]
    expect("write_csv nan mpix: no 'nan' in CSV row", "nan" not in row2, True)
    expect("write_csv nan mpix: no 'inf' in CSV row", "inf" not in row2, True)


def test_write_csv_zero_negative_guards():
    """Verify write_csv does not output 0.0 or negative mpix/pitch values."""
    section("write_csv zero/negative value guards")
    import tempfile
    import pixelpitch as pp
    from models import Spec

    # spec.mpix=0.0 → CSV should have empty mpix cell (not "0.0")
    spec_zero = Spec(name="Zero MP Cam", category="fixed", type=None,
                     size=(35.9, 23.9), pitch=5.12, mpix=0.0, year=2020)
    d_zero = pp.derive_spec(spec_zero)
    d_zero.id = 0
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        out_path = Path(f.name)
    try:
        pp.write_csv([d_zero], out_path)
        csv_text = out_path.read_text(encoding="utf-8")
    finally:
        out_path.unlink(missing_ok=True)
    row = csv_text.splitlines()[1]
    expect("write_csv zero mpix: no ',0.0,' in CSV row",
           ",0.0," not in row, True)

    # spec.mpix=-5.0 → CSV should have empty mpix cell (not "-5.0")
    spec_neg = Spec(name="Neg MP Cam", category="fixed", type=None,
                    size=(35.9, 23.9), pitch=5.12, mpix=-5.0, year=2020)
    d_neg = pp.derive_spec(spec_neg)
    d_neg.id = 0
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        out_path2 = Path(f.name)
    try:
        pp.write_csv([d_neg], out_path2)
        csv_text2 = out_path2.read_text(encoding="utf-8")
    finally:
        out_path2.unlink(missing_ok=True)
    row2 = csv_text2.splitlines()[1]
    expect("write_csv neg mpix: no '-5.0' in CSV row",
           "-5.0" not in row2, True)

    # derived.pitch=0.0 (direct spec.pitch=0.0 with no computed fallback)
    # → CSV should have empty pitch cell (not "0.00")
    spec_zero_pitch = Spec(name="Zero Pitch Cam", category="fixed", type=None,
                           size=(5.0, 3.7), pitch=0.0, mpix=0.0, year=2020)
    d_zero_pitch = pp.derive_spec(spec_zero_pitch)
    d_zero_pitch.id = 0
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        out_path3 = Path(f.name)
    try:
        pp.write_csv([d_zero_pitch], out_path3)
        csv_text3 = out_path3.read_text(encoding="utf-8")
    finally:
        out_path3.unlink(missing_ok=True)
    row3 = csv_text3.splitlines()[1]
    expect("write_csv zero pitch: no '0.00' pitch in CSV row",
           ",0.00," not in row3, True)


def main():
    test_imaging_resource()
    test_apotelyt()
    test_gsmarena()
    test_gsmarena_unicode_quotes()
    test_openmvg_csv_parser()
    test_openmvg_bom()
    test_merge_multi_source()
    test_csv_schema()
    test_parse_existing_csv()
    test_csv_round_trip()
    test_deduplicate_specs()
    test_merge_camera_data()
    test_sensor_size_from_type()
    test_parse_sensor_field()
    test_pixel_pitch()
    test_derive_spec_zero_pitch()
    test_match_sensors()
    test_load_sensors_database()
    test_load_csv()
    # test_cined_format_coverage removed — FORMAT_TO_MM dict removed from cined.py
    test_about_html_rendering()
    test_gsmarena_select_main_lens()
    test_gsmarena_decimal_mp()
    test_create_camera_key_year_mismatch()
    test_category_dedup()
    test_merge_field_preservation()
    test_merge_size_consistency()
    test_merge_gsmarena_measured_preserved()
    test_merge_year_change_log()
    test_sony_dsc_hyphen_normalisation()
    test_mpix_re_format_handling()
    test_openmvg_negative_dimensions()
    test_sorted_by_zero_values()
    test_template_zero_pitch_rendering()
    test_template_negative_pitch_rendering()
    test_parse_existing_csv_negative_values()
    test_derive_spec_negative_size()
    test_derive_spec_computed_zero_pitch()
    test_write_csv_nonfinite_guards()
    test_write_csv_zero_negative_guards()

    print("\n" + ("=" * 60))
    if _failures:
        print(f"FAILED: {len(_failures)} check(s)")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
