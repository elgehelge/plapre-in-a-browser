// BPE tokenizer for Danish text + the special audio/text tags.
//
// Loads the model repo's tokenizer.json via @huggingface/transformers and
// resolves the special token ids the pipeline needs. Mirrors the ids resolved
// in plapre/inference.py __init__.

import { PreTrainedTokenizer } from "@huggingface/transformers";
import { artifactUrl, ARTIFACTS } from "./assets.js";
import { MissingModelError } from "./types.js";

export interface SpecialTokens {
  textTag: number; // <text>
  audioTag: number; // <audio>
  audioTokenStart: number; // <audio_0>
  audioTokenEnd: number; // <audio_12799>
  eos: number;
}

export class PlapreTokenizer {
  private constructor(
    private readonly tok: PreTrainedTokenizer,
    readonly special: SpecialTokens,
  ) {}

  static async load(): Promise<PlapreTokenizer> {
    const res = await fetch(artifactUrl("tokenizer"));
    if (!res.ok) {
      throw new MissingModelError("tokenizer.json", ARTIFACTS.tokenizer.producedBy);
    }
    const tokenizerJSON = await res.json();
    // tokenizer_config is optional for raw encode; pass an empty config.
    const tok = new PreTrainedTokenizer(tokenizerJSON, {});

    const id = (t: string): number => {
      const v = tok.model.tokens_to_ids.get(t);
      if (v === undefined) throw new Error(`Special token not found: ${t}`);
      return v;
    };

    return new PlapreTokenizer(tok, {
      textTag: id("<text>"),
      audioTag: id("<audio>"),
      audioTokenStart: id("<audio_0>"),
      audioTokenEnd: id("<audio_12799>"),
      eos: tok.model.tokens_to_ids.get("<|endoftext|>") ?? -1,
    });
  }

  /** Encode without special tokens (matches reference: add_special_tokens=False). */
  encode(text: string): number[] {
    return this.tok.encode(text, { add_special_tokens: false });
  }

  /** Build the prompt ids: [<text>] + text_ids + [<audio>]. */
  buildPrompt(text: string): number[] {
    return [this.special.textTag, ...this.encode(text), this.special.audioTag];
  }
}
