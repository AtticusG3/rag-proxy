#!/usr/bin/env python3
"""Back-compat wrapper: pool scaling moved to scale_ingest_capacity.py.

Kept so existing systemd units (nomic-embed-scale.service) and operator muscle
memory keep working. All logic lives in scripts/scale_ingest_capacity.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scale_ingest_capacity import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
