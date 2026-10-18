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
  /**
   * Generation stop ids. Always contains <eos>; also contains </audio> when the
   * tokenizer has it (Plapre 3 ends generation "</audio> <eos>" and both are
   * stop tokens; the pico/nano tokenizers have no </audio>).
   */
  stops: readonly number[];
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

    // Plapre's generation stop token is "<eos>" (id 0 — AutoTokenizer's
    // eos_token_id), NOT SmolLM2's "<|endoftext|>", which this retrained
    // tokenizer does not even contain. Resolving the wrong tag silently
    // disabled the stop condition (ran to maxTokens). Verified against the
    // torch golden in conversion/validate_lm_golden.py.
    const eos = id("<eos>");
    // Plapre 3 terminates generation with "</audio>" followed by "<eos>", and
    // upstream treats BOTH as stop tokens. The tag only exists in that
    // tokenizer, so resolve it optionally instead of per-variant.
    const audioClose = tok.model.tokens_to_ids.get("</audio>");
    return new PlapreTokenizer(tok, {
      textTag: id("<text>"),
      audioTag: id("<audio>"),
      audioTokenStart: id("<audio_0>"),
      audioTokenEnd: id("<audio_12799>"),
      eos,
      stops: audioClose === undefined ? [eos] : [eos, audioClose],
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
