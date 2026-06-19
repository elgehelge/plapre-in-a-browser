import { describe, it, expect } from "vitest";
import { danishNumberToWords } from "./num2words-da.js";

// These tests pin the observable behavior of the Danish number speller. The
// linguistically irregular cases (vigesimal tens, unit-before-ten with "og")
// are asserted as confident Danish. A few forms (article on "hundrede"/"tusind"
// and the decimal "en" vs "et") are not yet reconciled with num2words(da); those
// are marked as parity-pending and lock the *current* output so a future change
// to match golden Python output is a visible, intentional diff.

describe("danishNumberToWords — units and teens", () => {
  it.each([
    ["0", "nul"],
    ["1", "en"],
    ["7", "syv"],
    ["10", "ti"],
    ["11", "elleve"],
    ["13", "tretten"],
    ["16", "seksten"],
    ["17", "sytten"],
    ["19", "nitten"],
  ])("%s -> %s", (input, expected) => {
    expect(danishNumberToWords(input)).toBe(expected);
  });
});

describe("danishNumberToWords — irregular tens and unit+og+ten", () => {
  it.each([
    ["20", "tyve"],
    ["21", "enogtyve"],
    ["25", "femogtyve"],
    ["30", "tredive"],
    ["42", "toogfyrre"],
    ["50", "halvtreds"],
    ["55", "femoghalvtreds"],
    ["60", "tres"],
    ["70", "halvfjerds"],
    ["80", "firs"],
    ["90", "halvfems"],
    ["99", "nioghalvfems"],
  ])("%s -> %s", (input, expected) => {
    expect(danishNumberToWords(input)).toBe(expected);
  });
});

describe("danishNumberToWords — hundreds and thousands", () => {
  it("joins hundreds and remainder with 'og'", () => {
    expect(danishNumberToWords("234")).toBe("to hundrede og fireogtredive");
  });

  it("treats a dot as a thousands separator (matching the reference)", () => {
    expect(danishNumberToWords("1.500")).toBe("en tusind fem hundrede");
  });

  it("spells large scaled numbers", () => {
    expect(danishNumberToWords("2000000")).toBe("to millioner");
  });
});

describe("danishNumberToWords — decimals and sign", () => {
  it("spells the fractional part digit by digit after 'komma'", () => {
    expect(danishNumberToWords("3,14")).toBe("tre komma en fire");
  });

  it("prefixes negatives with 'minus'", () => {
    expect(danishNumberToWords("-5")).toBe("minus fem");
  });

  it("returns the input unchanged when it is not a finite number", () => {
    expect(danishNumberToWords("not-a-number")).toBe("not-a-number");
  });
});

// PARITY-PENDING (see num2words-da.ts header): these forms are not yet confirmed
// against num2words(lang="da"). They lock current output so the eventual
// reconciliation is an explicit change, not a silent one.
describe("danishNumberToWords — parity-pending forms (locks current behavior)", () => {
  it("uses 'en' (not 'et') as the article on hundrede/tusind and in decimals", () => {
    expect(danishNumberToWords("100")).toBe("en hundrede");
    expect(danishNumberToWords("1000")).toBe("en tusind");
    expect(danishNumberToWords("2,1")).toBe("to komma en");
  });
});
