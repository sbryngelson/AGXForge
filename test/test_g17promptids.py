"""tools/g17promptids.py: an explicit prompt-id file, through the raw tokenizer, exactly the requested length.

Needs the InternLM2.5-1.8B-chat tokenizer at ~/models/internlm2_5-1_8b-chat (skipped without it). CPU only.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path.home() / "models" / "internlm2_5-1_8b-chat"
sys.path.insert(0, str(ROOT / "tools"))


@unittest.skipUnless((MODEL / "tokenizer.json").is_file(), "the InternLM2 tokenizer is not installed")
class PromptIds(unittest.TestCase):
    def test_raw_tokenizer_round_trips_a_retained_prompt(self):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        ids = json.loads((ROOT / "evidence/g17-q8-decode-v1/q8-decode-1k.json").read_text())["prompt_ids"]
        self.assertEqual(tok.encode(tok.decode(ids, skip_special_tokens=False), add_special_tokens=False).ids, ids)

    def test_exact_length_and_template(self):
        text = (ROOT / "docs" / "g17-tensorops-machine-model.md").read_text()[:40000]
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "t.txt", Path(d) / "ids.json"
            src.write_text(text)
            subprocess.run([sys.executable, str(ROOT / "tools" / "g17promptids.py"), str(src), "--instruction",
                            "Summarize the following section.", "--tokens", "512", "--out", str(out)],
                           check=True, capture_output=True)
            ids = json.loads(out.read_text())
            prov = json.loads(out.with_suffix(".provenance.json").read_text())
        self.assertEqual(len(ids), 512)
        self.assertEqual(ids[:2], [1, 92543])                  # <s>, <|im_start|>
        # the template's tail: <|im_end|>, newline, <|im_start|>, "assistant", newline
        self.assertEqual(ids[-6:], [92542, 364, 92543, 525, 11353, 364])
        self.assertEqual(prov["prompt_ids"], ids)


if __name__ == "__main__":
    unittest.main()
