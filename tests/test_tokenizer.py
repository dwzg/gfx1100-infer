#!/usr/bin/env python3
"""Compare gfxi's tokenizer with Hugging Face `tokenizers` on the model's original tokenizer.json.

usage: test_tokenizer.py <gfxi> <model.gguf>=<tokenizer.json> [...]
"""
import os, random, subprocess, sys, tempfile, unicodedata
from tokenizers import Tokenizer

CASES = [
    "Hello, world!", " leading space", "trailing space ", "  two  spaces  ", "\n", "\n\n", " \n \n", "\t\tindented\tcode",
    "It's they're we've I'm you'll he'd DON'T", "naïve café résumé Ångström", "日本語のテキスト。中文文本，测试！", "한국어 텍스트",
    "Привет, мир! Здравствуйте.", "مرحبا بالعالم", "שלום עולם", "हिन्दी पाठ का परीक्षण", "ไทยทดสอบ",
    "emoji 🙂🚀👨‍👩‍👧‍👦 🇩🇪 done", "numbers 1 22 333 4444 55555 3.14159 1,000,000 0x1F 1e-9", "a_b-c.d/e\\f:g;h|i",
    "def f(x):\n    return x ** 2  # square\n\nprint(f(3))\n", "#include <stdio.h>\nint main(void) { printf(\"%d\\n\", 42); return 0; }\r\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    "<think>\nreasoning\n</think>\n\nanswer", "URL: https://example.com/path?query=1&x=y#frag", "mixed123abc456 ABCdef XMLHttpRequest camelCaseWord",
    "!!!???...,,,;;;:::", "''''\"\"\"\"````", "    ", " nbsp emspace　ideographic", "é combining ǟ", "ǅ ǈ ǋ titlecase ᾈ",
    "x" * 300, " " * 70 + "end", "\n" * 9 + "end", "The quick brown fox jumps over the lazy dog. " * 12, "αβγδε ΑΒΓΔΕ ϑϒ", "½ ¾ ² ³ ¹ ① Ⅳ ٣ ४",
    "$100 €50 £20 ¥10 + − × ÷ = ≠ ≤ ≥ ∑ ∫ √", "end.\r\n\r\nnext", "tab\there", "",
]

def random_cases(n, seed=1234):
    rng = random.Random(seed)
    pools = [list(range(0x20, 0x7F)), list(range(0xA0, 0x250)), list(range(0x370, 0x400)), list(range(0x400, 0x500)),
             list(range(0x4E00, 0x4E80)), list(range(0x3040, 0x30A0)), [0x0A, 0x0D, 0x09, 0x20, 0x20, 0x20], list(range(0x1F600, 0x1F640)),
             list(range(0x300, 0x330)), list(range(0x2000, 0x2070)), list(range(0x30, 0x3A))]
    out = []
    for _ in range(n):
        s = "".join(chr(rng.choice(rng.choice(pools))) for _ in range(rng.randint(1, 60)))
        out.append(unicodedata.normalize("NFC", s))
    return out

def main():
    exe, failed = sys.argv[1], 0
    for spec in sys.argv[2:]:
        gguf_path, tok_path = spec.split("=", 1)
        ref = Tokenizer.from_file(tok_path)
        cases = [unicodedata.normalize("NFC", c) for c in CASES] + random_cases(400)
        bad = 0
        with tempfile.TemporaryDirectory() as tmp:
            for c in cases:
                path = os.path.join(tmp, "in.txt")
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(c)
                got = subprocess.run([exe, "tokenize", gguf_path, "@" + path], check=True, capture_output=True, text=True).stdout.split()
                want = [str(i) for i in ref.encode(c, add_special_tokens=False).ids]
                if got != want:
                    bad += 1
                    if bad <= 4:
                        print(f"  MISMATCH {c!r}\n    ours {' '.join(got)}\n    ref  {' '.join(want)}")
        print(f"{'FAIL' if bad else 'ok  '} {os.path.basename(gguf_path)}: {len(cases) - bad}/{len(cases)} strings tokenize identically")
        failed += bad > 0
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
