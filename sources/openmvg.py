"""
openMVG/CameraSensorSizeDatabase — MIT-licensed CSV.

https://github.com/openMVG/CameraSensorSizeDatabase

Provides exact sensor width/height (mm) and pixel dimensions for thousands
of consumer cameras. Pixel pitch is derived (sensor_width_mm / pixel_width).
Coverage skews toward 2010s consumer compacts; recent (2024+) models lag.
"""

from __future__ import annotations

import csv
import io
from typing import Optional

from . import Spec, http_get, normalise_name

CSV_URL = (
    "https://raw.githubusercontent.com/openMVG/CameraSensorSizeDatabase/"
    "master/sensor_database_detailed.csv"
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

        # The openMVG dataset has no body-type field; classify by sensor size:
        # full frame and larger / APS-C are usually interchangeable-lens, the
        # rest skew toward compacts. This is a best-effort guess.
        if size:
            if size[0] >= 20:
                category = "mirrorless"
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
