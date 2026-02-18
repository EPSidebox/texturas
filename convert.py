"""
Texturas — Asset Conversion Toolkit
=====================================
Prepares all static assets for browser deployment.

Subcommands:
  python convert.py glove     --input glove.6B.50d.txt --output ./assets/vectors/
  python convert.py wordnet   --output ./assets/wordnet/
  python convert.py lexicons  --nrc-emolex path --nrc-intensity path --nrc-vad path
                              --vader path --sentiwordnet path --output ./assets/lexicons/
  python convert.py all       (runs all with default paths)

Requirements:
  pip install nltk
  python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
"""

import argparse
import json
import struct
import sys
import os
from pathlib import Path
from collections import defaultdict


# ════════════════════════════════════════════
#  GLOVE CONVERTER
# ════════════════════════════════════════════

def convert_glove(input_path, output_dir, top_n=50000, min_len=2):
    """Convert GloVe text file → vocab.json + vectors.bin"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[glove] Reading {input_path}...")
    words, vectors, dim = [], [], None

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            # Skip Word2Vec-style header
            if i == 0 and len(parts) == 2:
                try:
                    int(parts[0]); int(parts[1])
                    print(f"  Skipping header: {parts[0]} words × {parts[1]}d")
                    continue
                except ValueError:
                    pass
            if len(parts) < 3:
                continue
            word = parts[0]
            if len(word) < min_len:
                continue
            try:
                vec = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            if dim is None:
                dim = len(vec)
                print(f"  Detected dimensionality: {dim}")
            elif len(vec) != dim:
                continue
            words.append(word)
            vectors.append(vec)
            if (i + 1) % 100000 == 0:
                print(f"  Processed {i+1:,} lines, kept {len(words):,}")
            if top_n and len(words) >= top_n:
                break

    print(f"  Final: {len(words):,} words × {dim}d")

    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"dim": dim, "vocab": {w: i for i, w in enumerate(words)}},
                  f, separators=(",", ":"))

    vectors_path = output_dir / "vectors.bin"
    with open(vectors_path, "wb") as f:
        for vec in vectors:
            f.write(struct.pack(f"{dim}f", *vec))

    _print_sizes(vocab_path, vectors_path)


# ════════════════════════════════════════════
#  WORDNET EXTRACTOR
# ════════════════════════════════════════════

def extract_wordnet(output_dir, synset_depth=2, vocab_file=None):
    """Extract lemmatizer, POS lookup, and synset graph from WordNet via NLTK."""
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        print("[wordnet] ERROR: nltk not installed. Run: pip install nltk")
        print('  Then: python -c "import nltk; nltk.download(\'wordnet\')"')
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional vocab filter
    vocab_filter = None
    if vocab_file:
        print(f"[wordnet] Loading vocab filter from {vocab_file}...")
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            vocab_filter = set(data.get("vocab", data).keys())
        print(f"  Filter: {len(vocab_filter):,} words")

    # ── Lemmatizer exceptions ──
    print("[wordnet] Extracting lemmatizer data...")
    pos_map = {"noun": "n", "verb": "v", "adj": "a", "adv": "r"}
    exceptions = {}
    for pos_name, pos_tag in pos_map.items():
        exc = {}
        # NLTK stores exceptions internally; access via _exception_map
        try:
            for lemma in wn.all_lemma_names(pos=pos_tag):
                for form in wn._morphy(lemma, pos_tag, check_exceptions=False):
                    if form != lemma:
                        exc[form] = lemma
            # Also get explicit exception list
            exception_map = getattr(wn, '_exception_map', None)
            if exception_map and pos_tag in exception_map:
                for form, lemmas in exception_map[pos_tag].items():
                    if lemmas:
                        exc[form] = lemmas[0]
        except Exception:
            pass
        exceptions[pos_tag] = exc
        print(f"  {pos_name}: {len(exc):,} exception forms")

    # Morphological suffix rules (WordNet's detachment rules)
    rules = {
        "n": [["s", ""], ["ses", "s"], ["ves", "f"], ["xes", "x"],
              ["zes", "z"], ["ches", "ch"], ["shes", "sh"],
              ["men", "man"], ["ies", "y"]],
        "v": [["s", ""], ["ies", "y"], ["es", "e"], ["es", ""],
              ["ed", "e"], ["ed", ""], ["ing", "e"], ["ing", ""]],
        "a": [["er", ""], ["est", ""], ["er", "e"], ["est", "e"]],
        "r": []
    }

    lemmatizer_path = output_dir / "lemmatizer.json"
    with open(lemmatizer_path, "w", encoding="utf-8") as f:
        json.dump({"exceptions": exceptions, "rules": rules},
                  f, separators=(",", ":"))
    print(f"  lemmatizer.json: {lemmatizer_path.stat().st_size / 1024:.0f} KB")

    # ── POS Lookup Table ──
    print("[wordnet] Building POS lookup table...")
    pos_counts = defaultdict(lambda: defaultdict(int))
    for synset in wn.all_synsets():
        pos = synset.pos()
        # Normalize adj satellite to adj
        if pos == "s":
            pos = "a"
        for lemma in synset.lemmas():
            name = lemma.name().lower().replace("_", " ")
            if " " in name:
                continue  # skip multi-word
            if vocab_filter and name not in vocab_filter:
                continue
            pos_counts[name][pos] += lemma.count() or 1

    pos_lookup = {}
    for word, counts in pos_counts.items():
        best_pos = max(counts, key=counts.get)
        pos_lookup[word] = best_pos

    pos_path = output_dir / "pos-lookup.json"
    with open(pos_path, "w", encoding="utf-8") as f:
        json.dump(pos_lookup, f, separators=(",", ":"))
    print(f"  pos-lookup.json: {len(pos_lookup):,} entries, "
          f"{pos_path.stat().st_size / 1024:.0f} KB")

    # ── Synset Relationships ──
    print(f"[wordnet] Extracting synset relationships (depth={synset_depth})...")
    synsets_data = {}

    def get_related(synset, depth):
        """Recursively collect hypernyms/hyponyms/meronyms to given depth."""
        if depth <= 0:
            return None
        result = {
            "definition": synset.definition(),
            "hypernyms": [],
            "hyponyms": [],
            "meronyms": []
        }
        for h in synset.hypernyms():
            names = [l.name().lower() for l in h.lemmas()]
            entry = {"synset": h.name(), "words": names}
            if depth > 1:
                sub = get_related(h, depth - 1)
                if sub:
                    entry["children"] = sub
            result["hypernyms"].append(entry)
        for h in synset.hyponyms():
            names = [l.name().lower() for l in h.lemmas()]
            entry = {"synset": h.name(), "words": names}
            if depth > 1:
                sub = get_related(h, depth - 1)
                if sub:
                    entry["children"] = sub
            result["hyponyms"].append(entry)
        for m in synset.part_meronyms() + synset.substance_meronyms():
            names = [l.name().lower() for l in m.lemmas()]
            result["meronyms"].append({"synset": m.name(), "words": names})
        return result

    processed = 0
    for word, best_pos in pos_lookup.items():
        word_synsets = wn.synsets(word, pos=best_pos)
        if not word_synsets:
            continue
        # Take top synset (most frequent sense)
        top = word_synsets[0]
        key = f"{word}#{best_pos}"
        related = get_related(top, synset_depth)
        if related:
            # Flatten for compact storage: just keep word lists
            compact = {
                "synset": top.name(),
                "definition": related["definition"],
                "hypernyms": _flatten_words(related.get("hypernyms", [])),
                "hyponyms": _flatten_words(related.get("hyponyms", [])),
                "meronyms": _flatten_words(related.get("meronyms", []))
            }
            # Only store if there's actual relationship data
            if compact["hypernyms"] or compact["hyponyms"] or compact["meronyms"]:
                synsets_data[key] = compact
                processed += 1

        if processed % 5000 == 0 and processed > 0:
            print(f"  Processed {processed:,} entries...")

    synsets_path = output_dir / "synsets.json"
    with open(synsets_path, "w", encoding="utf-8") as f:
        json.dump(synsets_data, f, separators=(",", ":"))
    print(f"  synsets.json: {len(synsets_data):,} entries, "
          f"{synsets_path.stat().st_size / (1024*1024):.1f} MB")


def _flatten_words(relation_list):
    """Flatten nested synset relationships to unique word lists."""
    words = set()
    for entry in relation_list:
        words.update(entry.get("words", []))
        if "children" in entry:
            for child_rel in ["hypernyms", "hyponyms", "meronyms"]:
                words.update(_flatten_words(
                    entry["children"].get(child_rel, [])))
    return sorted(words)


# ════════════════════════════════════════════
#  LEXICON CONVERTER
# ════════════════════════════════════════════

def convert_nrc_emolex(input_path, output_path):
    """NRC EmoLex: tab-separated word/emotion/association → JSON"""
    print(f"[lexicons] Converting NRC EmoLex from {input_path}...")
    data = defaultdict(dict)
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emotion, assoc = parts[0].lower(), parts[1], parts[2]
            try:
                data[word][emotion] = int(assoc)
            except ValueError:
                continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dict(data), f, separators=(",", ":"))
    print(f"  {len(data):,} entries → {output_path}")


def convert_nrc_intensity(input_path, output_path):
    """NRC Affect Intensity: tab-separated word/intensity/emotion → JSON"""
    print(f"[lexicons] Converting NRC Affect Intensity from {input_path}...")
    data = defaultdict(dict)
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            # Format varies: word \t score \t emotion OR word \t emotion \t score
            word = parts[0].lower()
            try:
                # Try word/score/emotion
                score, emotion = float(parts[1]), parts[2]
            except ValueError:
                try:
                    # Try word/emotion/score
                    emotion, score = parts[1], float(parts[2])
                except ValueError:
                    continue
            data[word][emotion] = round(score, 4)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dict(data), f, separators=(",", ":"))
    print(f"  {len(data):,} entries → {output_path}")


def convert_nrc_vad(input_path, output_path):
    """NRC VAD: tab-separated word/valence/arousal/dominance → JSON"""
    print(f"[lexicons] Converting NRC VAD from {input_path}...")
    data = {}
    with open(input_path, "r", encoding="utf-8") as f:
        header = True
        for line in f:
            parts = line.strip().split("\t")
            if header:
                header = False
                if parts[0].lower() == "word":
                    continue
            if len(parts) < 4:
                continue
            word = parts[0].lower()
            try:
                data[word] = {
                    "v": round(float(parts[1]), 4),
                    "a": round(float(parts[2]), 4),
                    "d": round(float(parts[3]), 4)
                }
            except (ValueError, IndexError):
                continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"  {len(data):,} entries → {output_path}")


def convert_vader(input_path, output_path):
    """VADER lexicon: tab-separated word/mean/std/scores → JSON"""
    print(f"[lexicons] Converting VADER from {input_path}...")
    data = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            try:
                data[word] = round(float(parts[1]), 4)
            except ValueError:
                continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"  {len(data):,} entries → {output_path}")


def convert_sentiwordnet(input_path, output_path):
    """SentiWordNet 3.0: tab-separated POS/ID/PosScore/NegScore/Terms/Gloss → JSON
    Output keyed by word#pos → { pos, neg, obj, synset }"""
    print(f"[lexicons] Converting SentiWordNet from {input_path}...")
    # Collect all synset entries, keep first (most frequent) sense per word#pos
    data = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            try:
                pos = parts[0].strip()
                pos_score = round(float(parts[2]), 4)
                neg_score = round(float(parts[3]), 4)
                obj_score = round(1.0 - pos_score - neg_score, 4)
                terms = parts[4]
            except (ValueError, IndexError):
                continue

            # Normalize POS: a/s → a (adjective satellite = adjective)
            if pos == "s":
                pos = "a"

            # Parse terms: "word#sense_number word2#sense_number"
            for term_entry in terms.split():
                parts2 = term_entry.split("#")
                if len(parts2) != 2:
                    continue
                word = parts2[0].lower().replace("_", " ")
                sense_num = int(parts2[1])
                key = f"{word}#{pos}"
                # Keep only first sense (most frequent) per word#pos
                if key not in data or sense_num < data[key].get("_sn", 999):
                    data[key] = {
                        "p": pos_score,
                        "n": neg_score,
                        "o": obj_score,
                        "_sn": sense_num
                    }

    # Remove internal sense number tracking
    for key in data:
        data[key].pop("_sn", None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"  {len(data):,} entries → {output_path}")


def convert_lexicons(args):
    """Convert all lexicons to JSON."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.nrc_emolex:
        convert_nrc_emolex(args.nrc_emolex, output_dir / "nrc-emolex.json")
    if args.nrc_intensity:
        convert_nrc_intensity(args.nrc_intensity, output_dir / "nrc-intensity.json")
    if args.nrc_vad:
        convert_nrc_vad(args.nrc_vad, output_dir / "nrc-vad.json")
    if args.vader:
        convert_vader(args.vader, output_dir / "vader.json")
    if args.sentiwordnet:
        convert_sentiwordnet(args.sentiwordnet, output_dir / "sentiwordnet.json")

    print(f"\n[lexicons] Done! Files written to {output_dir}/")
    _print_dir_sizes(output_dir)


# ════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════

def _print_sizes(*paths):
    total = 0
    for p in paths:
        p = Path(p)
        sz = p.stat().st_size
        total += sz
        print(f"  {p.name}: {sz / (1024*1024):.1f} MB")
    est_gz = total * 0.65
    print(f"  Estimated gzipped: ~{est_gz / (1024*1024):.1f} MB")


def _print_dir_sizes(d):
    d = Path(d)
    total = 0
    for f in sorted(d.glob("*.json")):
        sz = f.stat().st_size
        total += sz
        print(f"  {f.name}: {sz / 1024:.0f} KB")
    print(f"  Total: {total / 1024:.0f} KB (~{total * 0.4 / 1024:.0f} KB gzipped)")


# ════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Texturas — Asset Conversion Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert.py glove --input glove.6B.50d.txt --output ./assets/vectors/
  python convert.py wordnet --output ./assets/wordnet/
  python convert.py lexicons --nrc-emolex NRC-Emotion-Lexicon-Wordlevel-v0.92.txt \\
                             --vader vader_lexicon.txt \\
                             --sentiwordnet SentiWordNet_3.0.0.txt \\
                             --output ./assets/lexicons/
        """)
    sub = parser.add_subparsers(dest="command", required=True)

    # ── glove ──
    p_glove = sub.add_parser("glove", help="Convert GloVe text → binary")
    p_glove.add_argument("--input", required=True, help="GloVe .txt file")
    p_glove.add_argument("--output", default="./assets/vectors")
    p_glove.add_argument("--top", type=int, default=50000)
    p_glove.add_argument("--min-len", type=int, default=2)

    # ── wordnet ──
    p_wn = sub.add_parser("wordnet", help="Extract WordNet data (requires NLTK)")
    p_wn.add_argument("--output", default="./assets/wordnet")
    p_wn.add_argument("--depth", type=int, default=2,
                      help="Synset traversal depth (default: 2)")
    p_wn.add_argument("--vocab", default=None,
                      help="Optional vocab.json to filter synset extraction")

    # ── lexicons ──
    p_lex = sub.add_parser("lexicons", help="Convert sentiment lexicons → JSON")
    p_lex.add_argument("--nrc-emolex", default=None)
    p_lex.add_argument("--nrc-intensity", default=None)
    p_lex.add_argument("--nrc-vad", default=None)
    p_lex.add_argument("--vader", default=None)
    p_lex.add_argument("--sentiwordnet", default=None)
    p_lex.add_argument("--output", default="./assets/lexicons")

    # ── all ──
    sub.add_parser("all", help="Show instructions for full pipeline")

    args = parser.parse_args()

    if args.command == "glove":
        convert_glove(args.input, args.output, args.top, args.min_len)
    elif args.command == "wordnet":
        extract_wordnet(args.output, args.depth, args.vocab)
    elif args.command == "lexicons":
        convert_lexicons(args)
    elif args.command == "all":
        print("""
Full conversion pipeline:

1. Convert GloVe vectors:
   python convert.py glove --input glove.6B.50d.txt --output ./assets/vectors/

2. Extract WordNet data:
   python convert.py wordnet --output ./assets/wordnet/
   
   Optionally filter to GloVe vocabulary:
   python convert.py wordnet --output ./assets/wordnet/ --vocab ./assets/vectors/vocab.json

3. Convert sentiment lexicons:
   python convert.py lexicons \\
     --nrc-emolex    NRC-Emotion-Lexicon-Wordlevel-v0.92.txt \\
     --nrc-intensity NRC-Affect-Intensity-Lexicon.txt \\
     --nrc-vad       NRC-VAD-Lexicon.txt \\
     --vader         vader_lexicon.txt \\
     --sentiwordnet  SentiWordNet_3.0.0.txt \\
     --output ./assets/lexicons/

4. Deploy: copy ./assets/ to your GitHub Pages repo
   Set ASSET_BASE_URL in Texturas to your hosted path.

Source files:
  GloVe:         nlp.stanford.edu/data/glove.6B.zip
  NRC lexicons:  saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
  VADER:         github.com/cjhutto/vaderSentiment
  SentiWordNet:  github.com/aesuli/SentiWordNet
  WordNet:       via NLTK (pip install nltk; nltk.download('wordnet'))
""")


if __name__ == "__main__":
    main()
