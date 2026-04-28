"""
Phase B — Real Futures Curve Carry Module

Replaces the Phase 2 snapshot carry with a proper time-series carry signal
based on Gorton/Rouwenhorst (2006) and Koijen/Moskowitz/Pedersen/Vrugt (2013).

Key improvements over Phase 2:
    1. Rolling carry time-series — tracks term structure over time, not just today
    2. Carry momentum — trend in carry itself (improving backwardation = stronger long)
    3. Vol-adjusted carry — normalize by realized vol so low-vol assets don't dominate
    4. Carry quality score — confidence weighting based on data freshness + curve depth

The module is designed to work with LSEG curve snapshots stored daily.
Falls back gracefully to the old snapshot carry when history isn't available.

References:
    - Gorton & Rouwenhorst (2006), "Facts and Fantasies about Commodity Futures"
    - Koijen, Moskowitz, Pedersen & Vrugt (2013), "Carry" (JFE)
    - Szymanowska et al. (2014), "An Anatomy of Commodity Futures Risk Premia"
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from signal_engine.resilience import log
