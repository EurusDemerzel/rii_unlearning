#!/usr/bin/env python3
"""
prepare_muse.py — Build a MUSE-style long-form book-unlearning dataset.

Protocol (mirrors MUSE: teacher sees forget books, unlearn on retain books):
  forget  = Alice's Adventures in Wonderland  (public domain, Lewis Carroll, 1865)
  retain  = The Wonderful Wizard of Oz        (public domain, L. Frank Baum, 1900)

Both books are chunked with the LLaMA-2 tokenizer. Held-out eval chunks are
disjoint from training chunks so perplexity/MIA changes reflect true forgetting.

Outputs: data/muse/muse_data.pkl
  {
    "tr_forget": [str,...],   # Alice chunks for teacher FT + NegGrad
    "ev_forget": [str,...],   # Alice chunks held-out for eval
    "tr_retain": [str,...],   # Oz chunks for FineTune / Retrain
    "ev_retain": [str,...],   # Oz chunks held-out for eval
  }
"""
import os, pickle, random
import numpy as np
from transformers import AutoTokenizer

random.seed(42); np.random.seed(42)

DATA = os.path.join("data", "muse")
TOK_DIR = os.path.join("data", "models", "Llama-2-7b-hf")

CHUNK_TOK = 320          # tokens per chunk (<= MAX_SEQ)
MAX_BOOK_TOK = 60000     # cap each book's token budget for runtime


def chunk_text(path, tokenizer, max_tok=MAX_BOOK_TOK, chunk=CHUNK_TOK):
    raw = open(path, encoding="utf-8").read()
    ids = tokenizer.encode(raw)[:max_tok]
    n_full = len(ids) // chunk
    chunks = []
    for i in range(n_full):
        c = ids[i * chunk:(i + 1) * chunk]
        chunks.append(tokenizer.decode(c))
    return chunks, len(ids)


def main():
    os.makedirs(DATA, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    print("tokenizer:", tok.name_or_path)

    alice_chunks, n_alice = chunk_text(os.path.join(DATA, "alice.txt"), tok)
    oz_chunks, n_oz = chunk_text(os.path.join(DATA, "oz.txt"), tok)
    print(f"Alice: {n_alice} tokens -> {len(alice_chunks)} chunks")
    print(f"Oz   : {n_oz} tokens -> {len(oz_chunks)} chunks")

    random.shuffle(alice_chunks); random.shuffle(oz_chunks)

    n_ev_forget = 25
    n_ev_retain = 25
    n_tr_forget = 90
    n_tr_retain = 120

    ev_forget = alice_chunks[:n_ev_forget]
    tr_forget = alice_chunks[n_ev_forget:n_ev_forget + n_tr_forget]
    ev_retain = oz_chunks[:n_ev_retain]
    tr_retain = oz_chunks[n_ev_retain:n_ev_retain + n_tr_retain]

    assert len(tr_forget) >= 50 and len(tr_retain) >= 80, "not enough chunks"

    data = dict(tr_forget=tr_forget, ev_forget=ev_forget,
                tr_retain=tr_retain, ev_retain=ev_retain)
    with open(os.path.join(DATA, "muse_data.pkl"), "wb") as fh:
        pickle.dump(data, fh)

    print(f"saved {os.path.join(DATA, 'muse_data.pkl')}")
    for k, v in data.items():
        print(f"  {k}: {len(v)} chunks | sample len: {len(v[0])} chars")


if __name__ == "__main__":
    main()
