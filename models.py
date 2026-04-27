"""Shared dataclasses for the pixelpitch project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Spec:
    name: str
    category: str
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
    matched_sensors: Optional[List[str]] = None
    id: Optional[int] = None
