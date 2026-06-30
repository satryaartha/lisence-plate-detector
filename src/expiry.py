"""
Stage 7 - Expiry date parsing and validity check.

Indonesian plates print the registration validity as MM.YY (month over year) at
the bottom of the plate. This module extracts that MM/YY from the recognised
bottom-row characters and compares it to the current date to return a clear
boolean expiry status.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date


@dataclass
class ExpiryResult:
    raw_text: str               # cleaned bottom-row OCR text
    month: int | None
    year: int | None
    is_expired: bool | None     # None when the date could not be parsed
    reason: str


def _normalise_digits(text: str) -> str:
    """Keep only digits (OCR may insert dots/spaces/letters)."""
    return re.sub(r"[^0-9]", "", text)


def parse_expiry(bottom_text: str) -> tuple[int | None, int | None]:
    """Parse a MM.YY (or MMYY) string into (month, year).

    Accepts inputs like '10.27', '1027', '10 27'. Returns (None, None) if it
    cannot find a plausible month/year pair.
    """
    digits = _normalise_digits(bottom_text)
    # Expect exactly the last 4 digits to be MMYY.
    if len(digits) < 4:
        return None, None
    mmyy = digits[-4:]
    month = int(mmyy[:2])
    year_two = int(mmyy[2:])
    if not (1 <= month <= 12):
        return None, None
    year = 2000 + year_two  # plates use two-digit years (e.g. 27 -> 2027)
    return month, year


def check_expiry(bottom_text: str, today: date | None = None) -> ExpiryResult:
    """Determine whether a plate is expired given its bottom-row OCR text."""
    today = today or date.today()
    month, year = parse_expiry(bottom_text)

    if month is None or year is None:
        return ExpiryResult(
            raw_text=bottom_text, month=None, year=None,
            is_expired=None,
            reason="Could not parse a valid MM/YY expiry from the plate.",
        )

    # Registration is valid through the end of the printed month.
    expired = (year, month) < (today.year, today.month)
    reason = (
        f"Expiry {month:02d}/{year} is "
        f"{'before' if expired else 'on/after'} current "
        f"{today.month:02d}/{today.year}."
    )
    return ExpiryResult(
        raw_text=bottom_text, month=month, year=year,
        is_expired=expired, reason=reason,
    )
