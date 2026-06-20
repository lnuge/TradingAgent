"""Per-process session identity, shared across all layers for audit logging."""
from __future__ import annotations

import os
import time
import uuid

SESSION_ID = os.getenv("TRADING_SESSION_ID") or f"sess-{int(time.time())}-{uuid.uuid4().hex[:6]}"
