import { describe, it, expect } from "vitest";
import { normalizeText, splitSentences } from "./normalize.js";

describe("normalizeText", () => {
  it("collapses all whitespace to single spaces", () => {
    expect(normalizeText("Hej\n\n  med \t dig")).toBe("Hej med dig");
  });

  it("strips trailing '--' caption separators", () => {
    expect(normalizeText("Rigtig tekst --- billedtekst her")).toBe("Rigtig tekst");
  });

  it("replaces numbers with Danish words in context", () => {
    expect(normalizeText("Jeg har 2 katte")).toBe("Jeg har to katte");
  });

  it("trims surrounding whitespace", () => {
    expect(normalizeText("  hej  ")).toBe("hej");
  });
});

describe("splitSentences", () => {
  it("splits on sentence-ending punctuation followed by whitespace", () => {
    expect(splitSentences("Hej med dig. Hvordan går det?")).toEqual([
      "Hej med dig.",
      "Hvordan går det?",
    ]);
  });

  it("does not split when fewer than two word chars precede the period", () => {
    // "H.C." has a non-word char before each period, so the \w{2} guard holds.
    expect(splitSentences("H.C. Andersen kom forbi.")).toEqual([
      "H.C. Andersen kom forbi.",
    ]);
  });

  it("strips leading Danish dialogue dashes", () => {
    expect(splitSentences("- Hej, sagde han.")).toEqual(["Hej, sagde han."]);
  });

  // Danish words frequently end in æ/ø/å. Python's re `\w` is Unicode-aware, so
  // the reference splits these correctly; the split must therefore treat Danish
  // letters as word characters too.
  it("splits sentences whose last word ends in a Danish letter", () => {
    expect(splitSentences("Himlen er blå. Bilen er rød.")).toEqual([
      "Himlen er blå.",
      "Bilen er rød.",
    ]);
  });
});
