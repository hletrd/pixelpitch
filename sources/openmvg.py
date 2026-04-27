"""
openMVG/CameraSensorSizeDatabase — MIT-licensed CSV.

https://github.com/openMVG/CameraSensorSizeDatabase

Provides exact sensor width/height (mm) and pixel dimensions for thousands
of consumer cameras. Pixel pitch is derived (sensor_width_mm / pixel_width).
Coverage skews toward 2010s consumer compacts; recent (2024+) models lag.

Category note: The dataset has no body-type field, so category is assigned
by a heuristic. Large-sensor cameras (>= 20mm width) are classified as
interchangeable-lens; a name-based check distinguishes DSLRs from mirrorless
cameras. Cameras that don't match any DSLR pattern default to "mirrorless"
because the majority of modern interchangeable-lens cameras are mirrorless.
Any misclassifications may produce duplicate entries when the same camera
appears in Geizhals data with the correct category.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Optional

from . import Spec, http_get, normalise_name

CSV_URL = (
    "https://raw.githubusercontent.com/openMVG/CameraSensorSizeDatabase/"
    "master/sensor_database_detailed.csv"
)

# DSLR name patterns — used to classify interchangeable-lens cameras when the
# dataset has no body-type field. These patterns match the vast majority of
# DSLRs in the openMVG database (Canon EOS *D, Nikon D*, Pentax K-*, etc.).
_DSLR_NAME_RE = re.compile(
    r"\b("
    r"Canon\s+EOS[-\s]+\dD"       # Canon EOS 5D, 6D, 7D, 1D, EOS-1D, etc.
    r"|Canon\s+EOS[-\s]+\dDs"     # Canon EOS-1Ds
    r"|Nikon\s+D\d{1,4}"          # Nikon D850, D5, D500, etc.
    r"|Pentax\s+K[-\s]\d"         # Pentax K-1, K-3, etc.
    r"|Pentax\s+\d{1,2}D"         # Pentax 645D
    r"|Sigma\s+SD\d?"             # Sigma SD1, SD9, SD14, etc.
    r"|Sony\s+DSLR-A\d+"          # Sony DSLR-A900, A700, etc.
    r"|Samsung\s+NX\d{3}"         # Samsung NX300 (some were DSLR-style)
    r")\b",
    re.IGNORECASE,
)


def fetch(limit: Optional[int] = None) -> list[Spec]:
    body = http_get(CSV_URL)
    if body is None:
        return []

    reader = csv.DictReader(io.StringIO(body))
    specs: list[Spec] = []
    for i, row in enumerate(reader):
        if limit is not None and i >= limit:
            break

        maker = (row.get("CameraMaker") or "").strip()
        model = (row.get("CameraModel") or "").strip()
        if not maker or not model:
            continue

        try:
            sw = float(row["SensorWidth(mm)"])
            sh = float(row["SensorHeight(mm)"])
        except (KeyError, ValueError, TypeError):
            sw = sh = None

        try:
            pw = int(float(row["SensorWidth(pixels)"]))
            ph = int(float(row["SensorHeight(pixels)"]))
            mpix = round(pw * ph / 1_000_000, 1) if pw and ph else None
        except (KeyError, ValueError, TypeError):
            mpix = None

        # If both mm dims and pixel dims known, the area*mpix derivation in
        # pixelpitch.derive_spec will produce the correct pitch — no need to
        # set Spec.pitch directly here.
        size = (sw, sh) if sw and sh else None
        name = normalise_name(f"{maker} {model}")

        # The openMVG dataset has no body-type field; classify by sensor size
        # and name heuristics. Large-sensor cameras (>= 20mm width) are
        # interchangeable-lens; a name check distinguishes DSLRs from mirrorless.
        # The rest skew toward compacts.
        if size:
            if size[0] >= 20:
                category = "dslr" if _DSLR_NAME_RE.search(name) else "mirrorless"
            else:
                category = "fixed"
        else:
            category = "fixed"

        specs.append(Spec(
            name=name,
            category=category,
            type=None,
            size=size,
            pitch=None,
            mpix=mpix,
            year=None,
        ))

    return specs


if __name__ == "__main__":
    rows = fetch(limit=5)
    for r in rows:
        print(r)
    print(f"... total fetched: {len(rows)}")
