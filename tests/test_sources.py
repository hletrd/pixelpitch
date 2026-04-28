"""
Cross-source validation: fetch the same well-known cameras from every
source and assert their values agree, against published reference values.

References (manufacturer specs):
  Sony A7 IV       : 35.9 x 23.9 mm, 33.0 MP -> 5.12 µm
  Sony ZV-E10      : 23.5 x 15.6 mm, 24.2 MP -> 3.92 µm
  Nikon Z9         : 35.9 x 23.9 mm, 45.7 MP -> 4.35 µm
  Canon R5 / R5 C  : 36.0 x 24.0 mm, 45.0 MP -> 4.39 µm
  Fujifilm GFX100S : 43.8 x 32.9 mm, 102.0 MP -> 3.76 µm
  OM System OM-1   : 17.4 x 13.0 mm, 20.4 MP -> 3.36 µm

Run:   python -m tests.test_sources
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources.apotelyt import fetch_one as apotelyt_fetch
from sources.imaging_resource import fetch_one as ir_fetch_one, _spec_url
from sources.gsmarena import fetch_phone

REF = {
    "Sony A7 IV":       {"size": (35.9, 23.9), "mpix": 33.0, "pitch": 5.12, "tol": 0.10},
    "Sony ZV-E10":      {"size": (23.5, 15.6), "mpix": 24.2, "pitch": 3.92, "tol": 0.10},
    "Nikon Z9":         {"size": (35.9, 23.9), "mpix": 45.7, "pitch": 4.35, "tol": 0.10},
    "Canon R5":         {"size": (36.0, 24.0), "mpix": 45.0, "pitch": 4.39, "tol": 0.10},
    "Fujifilm GFX 100S": {"size": (43.8, 32.9), "mpix": 102.0, "pitch": 3.76, "tol": 0.10},
    "OM System OM-1":   {"size": (17.4, 13.0), "mpix": 20.4, "pitch": 3.36, "tol": 0.20},
}

IR_URLS = {
    "Sony A7 IV":        "https://www.imaging-resource.com/cameras/sony-a7-iv-review/",
    "Sony ZV-E10":       "https://www.imaging-resource.com/cameras/sony-zv-e10-review/",
    "Nikon Z9":          "https://www.imaging-resource.com/cameras/nikon-z9-review/",
    "Canon R5":          "https://www.imaging-resource.com/cameras/canon-r5-c-review/",  # R5 C uses same sensor as R5
    "Fujifilm GFX 100S": "https://www.imaging-resource.com/cameras/fujifilm-gfx-100s-review/",
    "OM System OM-1":    "https://www.imaging-resource.com/cameras/om-system-om-1-review/",
}

APOTELYT_URLS = {
    "Sony A7 IV":        "https://apotelyt.com/camera-specs/sony-a7-iv-sensor-pixels",
    "Sony ZV-E10":       "https://apotelyt.com/camera-specs/sony-zv-e10-sensor-pixels",
    "Nikon Z9":          "https://apotelyt.com/camera-specs/nikon-z9-sensor-pixels",
    "Canon R5":          "https://apotelyt.com/camera-specs/canon-r5-sensor-pixels",
    "Fujifilm GFX 100S": "https://apotelyt.com/camera-specs/fujifilm-gfx-100s-sensor-pixels",
    "OM System OM-1":    "https://apotelyt.com/camera-specs/om-system-om-1-sensor-pixels",
}


def within(a, b, tol):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def check(label, got, want, tol):
    ok = within(got, want, tol)
    mark = "OK" if ok else "FAIL"
    if got is None:
        return f"  [{mark}] {label}: missing (want {want})"
    return f"  [{mark}] {label}: got {got:.2f} (want {want:.2f}, ±{tol})"


def validate_camera_sources(name, ir_url, ap_url, ref):
    print(f"=== {name} ===  (ref size={ref['size']}, mpix={ref['mpix']}, pitch={ref['pitch']})")

    ir = ir_fetch_one(_spec_url(ir_url))
    if ir:
        print(check("IR width ", ir.size[0] if ir.size else None, ref["size"][0], 0.5))
        print(check("IR height", ir.size[1] if ir.size else None, ref["size"][1], 0.5))
        print(check("IR mpix  ", ir.mpix, ref["mpix"], 0.5))
        print(check("IR pitch ", ir.pitch, ref["pitch"], ref["tol"]))
    else:
        print("  [FAIL] IR: no data")

    ap = apotelyt_fetch(ap_url)
    if ap:
        print(check("AP width ", ap.size[0] if ap.size else None, ref["size"][0], 0.5))
        print(check("AP height", ap.size[1] if ap.size else None, ref["size"][1], 0.5))
        print(check("AP mpix  ", ap.mpix, ref["mpix"], 1.0))  # apotelyt rounds differently
        print(check("AP pitch ", ap.pitch, ref["pitch"], ref["tol"]))
    else:
        print("  [FAIL] Apotelyt: no data")
    print()


def validate_gsmarena():
    print("=== GSMArena smartphone sensor (Galaxy S25 Ultra) ===")
    s = fetch_phone("samsung_galaxy_s25_ultra-13322.php")
    print(s)
    # Reference: Samsung HP2 in S24/S25 Ultra: 1/1.3", ~9.84x7.4 mm, 200 MP, 0.6 µm
    if s:
        print(check("GSM mpix ", s.mpix, 200.0, 5.0))
        print(check("GSM pitch", s.pitch, 0.6, 0.05))
        print(check("GSM type ", 1.3 if s.type == "1/1.3" else 0, 1.3, 0.001))
    print()


def main():
    for name, ref in REF.items():
        validate_camera_sources(name, IR_URLS[name], APOTELYT_URLS[name], ref)
    validate_gsmarena()


if __name__ == "__main__":
    main()
