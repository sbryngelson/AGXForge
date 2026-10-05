#!/usr/bin/env python3
"""Token ids for a NEW decode comparison: a text, wrapped in InternLM2's chat template, cut to an exact length.

    python3 tools/g17promptids.py TEXT_FILE --instruction "Summarize the following." --tokens 1792 --out IDS.json \\
        [--model ~/models/internlm2_5-1_8b-chat]

The matched study (MM 25.211) ran at a 1,792-token prompt whose ids were not retained; its receipts describe the
prompt only in words (evidence/g17-matched-study-v1/inputs-recovery.json). This tool makes the prompt of a NEW run
explicit and repeatable instead: the user message is the instruction, a blank line and the text, trimmed from the end
of the text until the templated prompt is exactly --tokens long, so the ids file - not a description - is the input.

Writes OUT as the plain JSON list of ids, and OUT with the suffix .provenance.json as {"prompt_ids", "tokens",
"text_sha256", "instruction", "template", "tokenizer": {path, file sha256s}}. The ids go to tools/g17twin.py e2e (--ctx NAME=GRAPHDIR=IDS.json) and, as "prompt_ids", into the graph's model config so the
graph is built with the same prompt. CPU only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# InternLM2's chat template (chat_template.jinja beside the checkpoint), written out: one user turn, then the
# assistant header (add_generation_prompt). bos_token is <s>.
TEMPLATE = "<s><|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def ids_for(tok, instruction, text):
    """THE RAW TOKENIZER, NOT transformers.AutoTokenizer: under transformers 5.16 the checkpoint's InternLM2TokenizerFast
    splits words into single characters ("Hello" -> 5 ids), while tokenizers.Tokenizer on the same tokenizer.json gives
    the word pieces the research runs recorded (it round-trips evidence/g17-q8-decode-v1/q8-decode-1k.json's 1,024 ids
    exactly). Special tokens in the template string map to their ids (<s> 1, <|im_start|> 92543, <|im_end|> 92542)."""
    return tok.encode(TEMPLATE.format(content=instruction + "\n\n" + text), add_special_tokens=False).ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(Path.home() / "models" / "internlm2_5-1_8b-chat"))
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    probe = ids_for(tok, "", "")
    if probe[:2] != [1, 92543]:
        raise SystemExit("unexpected template ids %r: is %s an InternLM2 tokenizer?" % (probe[:4], a.model))
    text = Path(a.text).read_text()
    ids = ids_for(tok, a.instruction, text)
    if len(ids) < a.tokens:
        raise SystemExit("the templated text is %d tokens; --tokens %d needs a longer text" % (len(ids), a.tokens))
    # trim the TEXT (never the template) from its end until the prompt fits: the longest character prefix whose prompt
    # is <= --tokens (token counts grow with the prefix), then refuse anything but an exact length
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(ids_for(tok, a.instruction, text[:mid])) <= a.tokens:
            lo = mid
        else:
            hi = mid - 1
    ids = ids_for(tok, a.instruction, text[:lo])
    if len(ids) != a.tokens:
        raise SystemExit("no prefix of the text gives exactly %d tokens (closest %d); adjust the text or --tokens"
                         % (a.tokens, len(ids)))
    files = {f: sha256(Path(a.model) / f) for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
             if (Path(a.model) / f).is_file()}
    record = dict(prompt_ids=ids, tokens=len(ids), text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  characters_used=lo, instruction=a.instruction, template=TEMPLATE, tokenizer_library="tokenizers (raw tokenizer.json)",
                  tokenizer=dict(path=a.model, files=files))
    # OUT is the plain id list tools/g17twin.py e2e hands to mlx-lm; the provenance record goes beside it
    Path(a.out).write_text(json.dumps(ids) + "\n")
    prov = Path(a.out).with_suffix(".provenance.json")
    prov.write_text(json.dumps(record, indent=1) + "\n")
    print("wrote %s: %d tokens (%d of %d characters of the text), and %s" % (a.out, len(ids), lo, len(text), prov))
    return 0


if __name__ == "__main__":
    sys.exit(main())
