// Danish text normalization for TTS.
//
// Mirrors plapre/inference.py: _normalize_text / _normalize_numbers /
// _split_sentences. Keeping this faithful matters: the LM is fed the BPE
// tokenization of this output, so deviations change the audio (and break parity
// with golden fixtures).

import { danishNumberToWords } from "./num2words-da.js";

const NUMBER_RE = /\d+(?:[,.]\d+)?/g;

/** Replace numbers with Danish words (e.g. "2,1" -> "to komma en"). */
export function normalizeNumbers(text: string): string {
  return text.replace(NUMBER_RE, (m) => danishNumberToWords(m));
}

/** Normalize raw article/document text for TTS. */
export function normalizeText(text: string): string {
  // Remove trailing "--- caption" separators (DOTALL in the reference).
  let out = text.trim().replace(/\s*-{2,}[\s\S]*$/, "");
  // Collapse all whitespace to single spaces.
  out = out.replace(/\s+/g, " ");
  out = normalizeNumbers(out);
  return out.trim();
}

/**
 * Split into sentences. Split on sentence-ending punctuation preceded by 2+
 * word chars and followed by whitespace; strip leading Danish dialogue dashes.
 */
export function splitSentences(text: string): string[] {
  const normalized = normalizeText(text);
  // JS lookbehind supports the \w{2}[.!?] guard used in the reference.
  const parts = normalized.split(/(?<=\w{2}[.!?])\s+/);
  const result: string[] = [];
  for (const part of parts) {
    const cleaned = part.trim().replace(/^[-–—]\s+/, "");
    if (cleaned) result.push(cleaned);
  }
  return result;
}
