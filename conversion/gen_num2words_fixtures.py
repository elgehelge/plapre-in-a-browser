"""Generate golden Danish number-word fixtures from num2words(lang="da").

The browser port (web/src/pipeline/num2words-da.ts) must reproduce these exactly
so that normalized text — and therefore the LM's audio tokens — matches the
reference pipeline (plapre/inference.py:_normalize_numbers, which calls
num2words(float(raw.replace(",", ".")), lang="da")).

Run with the conversion venv:
    conversion/.venv/bin/python conversion/gen_num2words_fixtures.py

Writes web/src/pipeline/__fixtures__/num2words-da.json.
"""

import json
from pathlib import Path

from num2words import num2words


def da(value) -> str:
    return num2words(value, lang="da")


# Integer inputs: the regex feeds danishNumberToWords the plain decimal string.
integer_inputs = [str(n) for n in range(0, 2000)]
integer_inputs += [
    "2000", "2025", "3000", "10000", "12345", "20000", "21000", "99999",
    "100000", "100001", "123456", "200000", "999999",
    "1000000", "1000001", "1500000", "1234567", "2000000", "21000000",
    "1000000000", "2000000000",
]

# Decimal inputs: the reference parses these via float(raw.replace(",", ".")),
# so a dot is a decimal point (e.g. "1.500" -> 1.5), not a thousands separator.
decimal_inputs = [
    "2,1", "2.1", "3,14", "0,5", "1.500", "1,234", "10,0", "0,0",
    "100,25", "12,34", "0,05",
]

negative_inputs = ["-1", "-5", "-21", "-100", "-2,5"]


def expected(raw: str) -> str:
    return da(float(raw.replace(",", ".")))


cases = [[raw, expected(raw)] for raw in integer_inputs + decimal_inputs + negative_inputs]

out = Path(__file__).resolve().parents[1] / "web/src/pipeline/__fixtures__/num2words-da.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cases, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
print(f"wrote {len(cases)} cases to {out}")
