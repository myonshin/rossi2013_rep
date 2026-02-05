
from __future__ import annotations

import numpy as np

def tds_m(first_year: int, first_month: int, n_obs: int) -> np.ndarray:
    """Port of tds_m.m: decimal time year + (month-1)/12."""
    years = np.empty(n_obs, dtype=int)
    months = np.empty(n_obs, dtype=int)
    y, m = first_year, first_month
    years[0], months[0] = y, m
    for i in range(1, n_obs):
        m += 1
        if m > 12:
            m = 1
            y += 1
        years[i], months[i] = y, m
    return years + (months - 1) / 12.0

def tds_q(first_year: int, first_quarter: int, n_obs: int) -> np.ndarray:
    """Port of tds_q.m: decimal time year + (quarter-1)/4."""
    years = np.empty(n_obs, dtype=int)
    qs = np.empty(n_obs, dtype=int)
    y, q = first_year, first_quarter
    years[0], qs[0] = y, q
    for i in range(1, n_obs):
        q += 1
        if q > 4:
            q = 1
            y += 1
        years[i], qs[i] = y, q
    return years + (qs - 1) / 4.0

def calds_m_string(first_year: int, first_month: int, n_obs: int) -> list[str]:
    """Port of calds_m_string.m: strings like '1957:01'."""
    out = []
    y, m = first_year, first_month
    for i in range(n_obs):
        out.append(f"{y}:{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out

def calds_q_string(first_year: int, first_quarter: int, n_obs: int) -> list[str]:
    """Port of calds_q_string.m: strings like '1957:1'."""
    out = []
    y, q = first_year, first_quarter
    for i in range(n_obs):
        out.append(f"{y}:{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return out

def calds_a_string(first_year: int, n_obs: int) -> list[str]:
    """Replacement for missing calds_a_string.m: strings like '1957'."""
    return [str(first_year + i) for i in range(n_obs)]
