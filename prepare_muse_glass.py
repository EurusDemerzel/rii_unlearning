#!/usr/bin/env python3
"""
prepare_muse_glass.py — Build the second MUSE-style retain variant.

forget = Alice's Adventures in Wonderland (Carroll)
retain = Through the Looking-Glass (Carroll)  -- same author, comparable style
         (vs. oz variant: different author)

Outputs: data/muse/muse_data_glass.pkl
  { "tr_retain": [...], "ev_retain": [...] }  (Oz/alice fields are shared)
"""
import os, pickle, random
import numpy as np
from transformers import AutoTokenizer

random.seed(42); np.random.seed(42)

DATA = os.path.join("data", "muse")
TOK_DIR = os.path.join("data", "models", "Llama-2-7b-hf")
CHUNK_TOK = 320
MAX_BOOK_TOK = 60000


def chunk_text(path, tokenizer, max_tok=MAX_BOOK_TOK, chunk=CHUNK_TOK):
    raw = open(path, encoding="utf-8").read()
    ids = tokenizer.encode(raw)[:max_tok]
    n_full = len(ids) // chunk
    return [tokenizer.decode(ids[i * chunk:(i + 1) * chunk])
            for i in range(n_full)], len(ids)


def main():
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    glass_chunks, n = chunk_text(os.path.join(DATA, "glass.txt"), tok)
    random.shuffle(glass_chunks)
    n_ev, n_tr = 25, 120
    data = dict(ev_retain=glass_chunks[:n_ev], tr_retain=glass_chunks[n_ev:n_ev + n_tr])
    with open(os.path.join(DATA, "muse_data_glass.pkl"), "wb") as fh:
        pickle.dump(data, fh)
    print(f"Looking-Glass: {n} tokens -> {len(glass_chunks)} chunks; "
          f"ev={len(data['ev_retain'])} tr={len(data['tr_retain'])}")
    print(f"saved {os.path.join(DATA, 'muse_data_glass.pkl')}")


if __name__ == "__main__":
    main()
