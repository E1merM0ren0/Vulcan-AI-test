#!/usr/bin/env python3
"""stream_dataset.py - build vulcan byte-LM .bin files from a Hugging Face dataset.

Format contract (must match vulcan/src/context.cpp loadBin/saveBin and the
vulcan_bitnet_gpt trainer):

    header : uint32 n, uint32 feat            (feat == 784)
    then   : n * feat float32 bytes of X      (each float is a raw byte 0..255)
    then   : n     float32 labels of y         (class id, used as 256 + y)

The model is a *byte* language model: each 784-token row is the UTF-8 bytes of a
text sample (padded right with 0 bytes to exactly BYTES). The optional label
column drives the classification head; with no label column every row gets 0.

Because storage is tight, this always uses HF streaming and an explicit cap;
it never bulk-downloads a dataset.

Usage:
    python tools/stream_dataset.py <repo> <base> [--split-test M] [flags]

Examples:
    # Starcoder python, first 1000 rows, text in "content", no label
    python tools/stream_dataset.py "bigcode/starcoderdata" code \
        --dataset-kwarg data_dir=python --text-col content --take 1000 --split-test 200

    # Chat distill, one message-turn text + category label
    python tools/stream_dataset.py "WithinUsAI/claude_mythos_distilled_25k" mythos \
        --messages --label-col category --take 1000 --split-test 100

Outputs (in vulcan/data/ by default):
    <base>_train.bin
    <base>_test.bin
"""
import argparse
import os
import struct


BYTES = 784  # must match BYTES in vulcan/src/bitnet_gpt.cpp


def text_from_row(row, messages, text_col):
    """Extract plain text to tokenize from a dataset row."""
    if text_col:
        if not isinstance(row.get(text_col), str):
            raise KeyError("--text-col %r not a string column" % text_col)
        return row[text_col]
    if messages:
        parts = row.get("messages") or row.get("conversations") or []
        out = []
        for turn in parts:
            c = (turn or {}).get("content")
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, list):
                out.append(" ".join(str(b.get("text", "")) for b in c if isinstance(b, dict)))
        return "\n".join(out)
    # standard text-style columns, in order of preference
    for col in ("content", "text", "prompt", "completion", "code", "answer"):
        if col in row and isinstance(row[col], str):
            return row[col]
    if "prompt" in row and "completion" in row:
        return str(row["prompt"]) + "\n" + str(row["completion"])
    raise KeyError(
        "no usable text column; pass --text-col NAME or --messages "
        "(available: %s)" % ", ".join(sorted(k for k in row if isinstance(row[k], str)))
    )


def label_from_row(row, label_col):
    """Return an integer class id for the row (0 if no label column)."""
    if not label_col:
        return 0
    v = row[label_col]
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    return hash(str(v)) & 0x7FFFFFFF


def write_bin(path, X, y, feat=BYTES):
    n = len(y)
    with open(path, "wb") as f:
        f.write(struct.pack("II", n, feat))
        for row in X:
            assert len(row) == feat, "row width %d != feat %d" % (len(row), feat)
            f.write(struct.pack("%df" % feat, *row))
        f.write(struct.pack("%df" % n, *y))
    print("wrote %s  (n=%d feat=%d)" % (path, n, feat))


def main():
    p = argparse.ArgumentParser(description="Build vulcan byte-LM .bin files from an HF dataset.")
    p.add_argument("repo", help="Hugging Face dataset id, e.g. bigcode/starcoderdata")
    p.add_argument("base", help="output base name -> <base>_train.bin / _test.bin")
    p.add_argument("--text-col", default=None, help="text column name (auto-detect if omitted)")
    p.add_argument("--label-col", default=None, help="label/class column (optional, default 0)")
    p.add_argument("--messages", action="store_true",
                   help="use 'messages'/'conversations' chat turns as text")
    p.add_argument("--take", type=int, default=1000, help="max rows to consume (default 1000)")
    p.add_argument("--split-test", type=int, default=0,
                   help="how many of the taken rows go to _test (default 0 -> all train)")
    p.add_argument("--out-dir", default="vulcan/data", help="output directory")
    p.add_argument("--dataset-kwarg", action="append", default=[],
                   metavar="KEY=VALUE", help="extra load_dataset kwarg, e.g. data_dir=python")
    p.add_argument("--split", default="train", help="dataset split (default train)")
    args = p.parse_args()

    from datasets import load_dataset  # imported lazily so --help works without datasets

    kwargs = {"streaming": True, "split": args.split}
    for kv in args.dataset_kwarg:
        k, _, v = kv.partition("=")
        if v.strip() in ("true", "false", "True", "False"):
            kwargs[k] = v.strip().lower() == "true"
        else:
            try:
                kwargs[k] = int(v) if "." not in v else float(v)
            except ValueError:
                kwargs[k] = v

    print("loading %s (streaming) take=%d ..." % (args.repo, args.take))
    ds = load_dataset(args.repo, **kwargs)

    # single pass: keep text + raw label per row (capped at --take, so memory-safe)
    texts, raw_labels = [], []
    raw_val = None
    for _, row in zip(range(args.take), iter(ds)):
        text = text_from_row(row, args.messages, args.text_col)
        if not text:
            continue
        raw_val = label_from_row(row, args.label_col) if args.label_col else 0
        texts.append(text)
        raw_labels.append(raw_val)
    print("consumed %d non-empty rows" % len(texts))

    # normalize labels to a dense contiguous range if a label column was given
    if args.label_col:
        uniq = sorted(set(raw_labels))
        lab_map = {v: k for k, v in enumerate(uniq)}
        labels = [lab_map[v] for v in raw_labels]
        n_cls = len(uniq)
        print("classes: %d" % n_cls)
    else:
        labels = [0] * len(texts)
        n_cls = 1

    X = []
    for text in texts:
        x = list(text.encode("utf-8", errors="replace")[:BYTES])
        if len(x) < BYTES:
            x = x + [0] * (BYTES - len(x))
        X.append(x)

    os.makedirs(args.out_dir, exist_ok=True)
    split_test = min(args.split_test, len(texts))
    train_end = len(texts) - split_test
    if split_test:
        parts = (("train", slice(0, train_end)), ("test", slice(train_end, len(texts))))
    else:
        parts = (("train", slice(0, len(texts))),)

    for name, sl in parts:
        idxs = list(range(*sl.indices(len(texts))))
        if not idxs:
            continue
        Xs = [X[i] for i in idxs]
        ys = [labels[i] for i in idxs]
        path = os.path.join(args.out_dir, "%s_%s.bin" % (args.base, name))
        write_bin(path, Xs, ys)
    print("done. base=%s ncls=%d" % (args.base, n_cls))


if __name__ == "__main__":
    main()
