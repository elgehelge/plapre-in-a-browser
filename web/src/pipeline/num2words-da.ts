// Danish cardinal number -> words.
//
// Mirrors the intent of `num2words(..., lang="da")` used by the reference
// pipeline (plapre/inference.py:_normalize_numbers). Danish numerals are
// irregular (vigesimal-ish tens: halvtreds=50, tres=60, halvfjerds=70,
// firs=80, halvfems=90) and units precede tens joined by "og" (enogtyve=21).
//
// PARITY CAVEAT: exact string parity with the `num2words` package has not yet
// been locked. To reproduce golden LM token ids, this must be reconciled with
// num2words(da) using fixtures (see docs/PLAN.md, Phase 2). Integer cardinals
// below are believed correct; decimal formatting ("komma" + digit-by-digit) is
// a defensible default that needs confirmation against num2words.

const ONES = [
  "nul", "en", "to", "tre", "fire", "fem", "seks", "syv", "otte", "ni",
  "ti", "elleve", "tolv", "tretten", "fjorten", "femten", "seksten",
  "sytten", "atten", "nitten",
];

const TENS: Record<number, string> = {
  20: "tyve",
  30: "tredive",
  40: "fyrre",
  50: "halvtreds",
  60: "tres",
  70: "halvfjerds",
  80: "firs",
  90: "halvfems",
};

function under100(n: number): string {
  if (n < 20) return ONES[n];
  const tens = Math.floor(n / 10) * 10;
  const unit = n % 10;
  if (unit === 0) return TENS[tens];
  return `${ONES[unit]}og${TENS[tens]}`;
}

function under1000(n: number): string {
  if (n < 100) return under100(n);
  const hundreds = Math.floor(n / 100);
  const rest = n % 100;
  const head = `${ONES[hundreds]} hundrede`;
  return rest === 0 ? head : `${head} og ${under100(rest)}`;
}

function integerToWords(n: number): string {
  if (n === 0) return "nul";
  const parts: string[] = [];
  const scales: [number, string, string][] = [
    [1_000_000_000, "milliard", "milliarder"],
    [1_000_000, "million", "millioner"],
    [1_000, "tusind", "tusind"],
  ];
  let remaining = n;
  for (const [value, singular, plural] of scales) {
    if (remaining >= value) {
      const count = Math.floor(remaining / value);
      remaining = remaining % value;
      const word = under1000(count);
      parts.push(`${word} ${count === 1 ? singular : plural}`);
    }
  }
  if (remaining > 0) parts.push(under1000(remaining));
  return parts.join(" ");
}

/** Convert a numeric string ("2", "2,1", "1.500") to Danish words. */
export function danishNumberToWords(raw: string): string {
  const normalized = raw.replace(/\./g, "").replace(",", ".");
  const value = Number(normalized);
  if (!Number.isFinite(value)) return raw;

  const negative = value < 0;
  const abs = Math.abs(value);
  const [intPart, fracPart] = String(abs).split(".");

  let words = integerToWords(Number(intPart));
  if (fracPart) {
    const digits = fracPart.split("").map((d) => ONES[Number(d)]).join(" ");
    words = `${words} komma ${digits}`;
  }
  return negative ? `minus ${words}` : words;
}
