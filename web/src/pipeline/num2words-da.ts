// Danish cardinal number -> words, reproducing num2words(lang="da").
//
// The reference pipeline normalizes numbers with
//   num2words(float(raw.replace(",", ".")), lang="da")
// (plapre/inference.py:_normalize_numbers). To keep the LM's audio tokens in
// parity with the reference we must match num2words exactly, including its
// quirks: "60" -> "treds" (not the standard "tres"), hundreds/thousands written
// without spaces ("ethundrede", "ettusind"), millions/milliards always plural
// with "en" ("en millioner"), and a dot parsed as a decimal point so "1.500"
// means 1.5. Parity is locked by the golden fixture in __fixtures__ (generated
// by conversion/gen_num2words_fixtures.py).

// Standalone digit / unit words; 1 is "et" everywhere except the "enog…" tens
// compound (see UNIT_IN_TENS).
const ONES = [
  "nul", "et", "to", "tre", "fire", "fem", "seks", "syv", "otte", "ni",
  "ti", "elleve", "tolv", "tretten", "fjorten", "femten", "seksten",
  "sytten", "atten", "nitten",
];

const UNIT_IN_TENS = ["", "en", "to", "tre", "fire", "fem", "seks", "syv", "otte", "ni"];

const TENS: Record<number, string> = {
  20: "tyve",
  30: "tredive",
  40: "fyrre",
  50: "halvtreds",
  60: "treds",
  70: "halvfjerds",
  80: "firs",
  90: "halvfems",
};

function under100(n: number): string {
  if (n < 20) return ONES[n];
  const tens = Math.floor(n / 10) * 10;
  const unit = n % 10;
  return unit === 0 ? TENS[tens] : `${UNIT_IN_TENS[unit]}og${TENS[tens]}`;
}

function under1000(n: number): string {
  if (n < 100) return under100(n);
  const hundreds = Math.floor(n / 100);
  const rest = n % 100;
  const head = `${ONES[hundreds]}hundrede`;
  return rest === 0 ? head : `${head} og ${under100(rest)}`;
}

function underMillion(n: number): string {
  if (n < 1000) return under1000(n);
  const thousands = Math.floor(n / 1000);
  const rest = n % 1000;
  const count = under1000(thousands);
  if (rest === 0) return `${count}tusind`;
  // num2words normally joins with the long form "tusinde og <rest>", but drops
  // the "og" and uses "tusind" concatenated only when both the thousands count
  // and the remainder are >= 100 (e.g. 123456 -> "…treogtyvetusindfirehundrede…").
  return thousands >= 100 && rest >= 100
    ? `${count}tusind${under1000(rest)}`
    : `${count}tusinde og ${under1000(rest)}`;
}

function scaled(n: number, divisor: number, word: string, below: (n: number) => string): string {
  const count = Math.floor(n / divisor);
  const rest = n % divisor;
  // Million/milliard counts use "en" for 1 (not "et") and the word stays plural.
  const head = `${count === 1 ? "en" : under1000(count)} ${word}`;
  return rest === 0 ? head : `${head} ${below(rest)}`;
}

function integerToWords(n: number): string {
  if (n >= 1_000_000_000) return scaled(n, 1_000_000_000, "milliarder", (r) => integerToWords(r));
  if (n >= 1_000_000) return scaled(n, 1_000_000, "millioner", underMillion);
  return underMillion(n);
}

/** Convert a numeric string ("2", "2,1", "1.500") to Danish words. */
export function danishNumberToWords(raw: string): string {
  // Mirror the reference's float() parse: comma -> decimal point, and a dot is
  // a decimal point too (so "1.500" is 1.5), never a thousands separator.
  const value = Number(raw.replace(",", "."));
  if (!Number.isFinite(value)) return raw;

  const negative = value < 0;
  const abs = Math.abs(value);

  let words: string;
  if (Number.isInteger(abs)) {
    words = integerToWords(abs);
  } else {
    const [intPart, fracPart] = String(abs).split(".");
    const digits = fracPart.split("").map((d) => ONES[Number(d)]).join(" ");
    words = `${integerToWords(Number(intPart))} komma ${digits}`;
  }
  return negative ? `minus ${words}` : words;
}
