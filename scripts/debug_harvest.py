"""Debug: show what _harvest_xml_record produces for OBJ-001."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_xml import _harvest_xml_record

xml_path = (
    Path(__file__).resolve().parent.parent / "data" / "xml" / "objects" / "OBJ-001.xml"
)
with open(xml_path) as f:
    record = _harvest_xml_record(f.read(), root="object")

print(json.dumps(record, indent=2, default=str))

# Also check OBJ-003 (empty constituents)
xml_path3 = (
    Path(__file__).resolve().parent.parent / "data" / "xml" / "objects" / "OBJ-003.xml"
)
with open(xml_path3) as f:
    record3 = _harvest_xml_record(f.read(), root="object")

print("\n--- OBJ-003 ---")
print(json.dumps(record3, indent=2, default=str))

# Also check terminology
term_path = Path(__file__).resolve().parent.parent / "data" / "xml" / "terminology.xml"
with open(term_path) as f:
    term = _harvest_xml_record(f.read(), root="terminology")

print("\n--- terminology ---")
print(json.dumps(term, indent=2, default=str))
