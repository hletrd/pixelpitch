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

Known heuristic limitations: The DSLR regex covers major brands (Canon EOS
xD/xxD/xxxD, Nikon D, Pentax K, Sigma SD, Sony DSLR-A) but may miss
obscure DSLR brands or unusual naming patterns. Any misclassifications may
produce duplicate entries when the same camera appears in Geizhals data with
the correct category.
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
# Note: this heuristic is imperfect — some DSLRs may be missed (e.g., obscure
# brands) and non-DSLRs should not be included (e.g., Samsung NX is mirrorless).
_DSLR_NAME_RE = re.compile(
    r"\b("
    r"Canon\s+EOS[-\s]+\d+D"     # Canon EOS 5D, 6D, 7D, 1D, 250D, 800D, 850D, etc.
    r"|Canon\s+EOS[-\s]+\d+Ds"   # Canon EOS-1Ds
    r"|Nikon\s+D\d{1,4}"          # Nikon D850, D5, D500, etc.
    r"|Pentax\s+K[-\s]?\d+[A-Za-z]*"  # Pentax K-1, K3, KP, KF, K-r, K100D, etc.
    r"|Pentax\s+\d{1,3}[DZ]"     # Pentax 645D, 645Z
    r"|Sigma\s+SD\d+"            # Sigma SD1, SD9, SD10, SD14, SD15, etc.
    r"|Sony\s+DSLR-A\d+"          # Sony DSLR-A900, A700, etc.
    r")\b",
    re.IGNORECASE,
)


def fetch(limit: Optional[int] = None) -> list[Spec]:
    body = http_get(CSV_URL)
    if body is None:
        return []

    # Strip UTF-8 BOM if present. If the upstream CSV is saved with a BOM
    # (e.g., by Excel), DictReader would produce mangled field names like
    # "﻿CameraMaker" instead of "CameraMaker", causing KeyError on
    # every row and 0 records returned.
    if body and body[0] == "﻿":
        body = body[1:]

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
