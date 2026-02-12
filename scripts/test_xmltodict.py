"""Quick check of xmltodict output shapes for object + terminology XML."""

import json
from pathlib import Path

import xmltodict

DATA_DIR = Path(__file__).parent.parent / "data"

# terminology
with open(DATA_DIR / "xml" / "terminology.xml") as f:
    raw = xmltodict.parse(f.read(), force_list=("term",))
print("=== terminology ===")
print(json.dumps(raw, indent=2))

# single-constituent object (OBJ-002)
with open(DATA_DIR / "xml" / "objects" / "OBJ-002.xml") as f:
    raw = xmltodict.parse(
        f.read(),
        force_list=("constituent", "classification", "dimension", "image"),
    )
print("\n=== OBJ-002 ===")
print(json.dumps(raw, indent=2))

# empty-constituents object (OBJ-003)
with open(DATA_DIR / "xml" / "objects" / "OBJ-003.xml") as f:
    raw = xmltodict.parse(
        f.read(),
        force_list=("constituent", "classification", "dimension", "image"),
    )
print("\n=== OBJ-003 ===")
print(json.dumps(raw, indent=2))
