# ⬡ Textures

**Multi-layered correlated textual analysis**

Textures is a browser-based tool for identifying thematic, discursive, and affective patterns in text. It layers corpus linguistics, multi-lexicon sentiment analysis, spreading activation relevance scoring, and network modularity onto the same text — and cross-correlates these layers to surface patterns no single method reveals alone.

From Latin *textura* (weaving) — the root of "text." Plural: the many woven layers.

**Author:** Ernesto Peña · Northeastern University  
**Version:** 0.8.1  
**Live:** [epsidebox.github.io/texturas/vellum/](https://epsidebox.github.io/texturas/vellum/)

---

## How It Works

Textures runs entirely in the browser. No server, no API keys. Static data assets (sentiment lexicons, WordNet data) are loaded once and cached locally via IndexedDB for instant subsequent visits.

A single unified analysis pipeline feeds both visualization modes:

```
Text input (paste / upload .txt / import .xml / separator markers)
  → Smart quote normalization
  → Paragraph-preserving tokenization (regex-based)
  → POS tagging (40K lookup table + suffix rules, 4-way: n/v/a/r)
  → Lemmatization (WordNet exception lists + morphological rules)
  → Frequency analysis (unigrams, bigrams, trigrams, stop-word filtered)
  → Multi-lexicon sentiment scoring (5 lexicons, POS-informed, sense-aware)
  → Spreading activation relevance (WordNet graph traversal with decay)
  → Co-occurrence matrix + Louvain community detection
  → Two output views from one pass:
      → Vellum: flat enriched tokens → grid binning → Three.js 3D visualization
      → Weave: paragraph-structured tokens → annotated text reader
  → Export: TEI XML (inline/standoff), CSV, Markdown report
```

---

## Features

### Visualization Modes

**Vellum** — 3D grid visualization (Three.js)
- Text mapped to a configurable grid (10², 20², 30²)
- **Channels page:** size = relevance, color = VADER polarity, height = arousal
- **Emotions page:** 3×3 Plutchik sub-boxes per cell, brightness = emotion proportion
- Isometric/flat camera transitions, cell hover/click-to-pin tooltips
- Multi-document patchwork view (Ctrl+click)
- Word panel with click-to-filter

**Weave** — Annotated text reader
- Six visual layers projected onto the text itself:
  - Polarity → text color (green/red/gray)
  - Emotion → stacked underlines (Plutchik colors, toggleable per emotion)
  - Arousal → underline thickness
  - Frequency → brightness (opacity)
  - Relevance → font weight (Roboto Mono 100–700)
  - Community → background highlight
- Minimap with click/drag navigation
- Word panel with click-to-highlight

### Analysis

- **Multi-lexicon sentiment** — NRC EmoLex (Plutchik 8 emotions), NRC Affect Intensity, NRC VAD (valence/arousal/dominance), VADER (polarity), SentiWordNet (sense-level pos/neg/obj)
- **Spreading activation** — WordNet-based relevance scoring with configurable depth, decay, and directional flow (bidirectional, upward/hypernyms, downward/hyponyms)
- **Community detection** — Louvain modularity on windowed co-occurrence of top-N terms
- **Multi-document support** — per-document analysis with shared parameters for comparability

### Export

- **TEI XML** — inline and standoff annotation modes, per-document and corpus-level (`<teiCorpus>`)
- **CSV** — per-token, all annotation layers as columns, layer-filtered by checkboxes
- **Markdown** — summary report with parameters, top words, communities, citations
- **TEI import** — parse `<w>` elements from Textures TEI output, reconstruct text for re-analysis

---

## Technical Architecture

- **Platform:** Static web app, GitHub Pages deployment
- **Runtime:** React 18 + Babel standalone (JSX compiled in browser, no build toolchain)
- **3D:** Three.js r128
- **Utilities:** Lodash 4, SheetJS 0.18.5 (XLSX export)
- **Font:** Roboto Mono (variable weight 100–700)
- **NLP:** All client-side — factory function engines for POS, lemmatization, synset traversal, sentiment scoring
- **Caching:** IndexedDB for loaded assets (~11MB first visit, instant thereafter)

### File Structure

```
vellum/
├── index.html      ← HTML wrapper (CDN deps + Babel + app.jsx)
├── app.jsx         ← Full application (unified pipeline + Vellum + Weave + Output + About)
```

### Deployed Assets

```
assets/
├── vectors/
│   ├── vocab.json          (~2 MB, 50K words)
│   └── vectors.bin         (~10 MB, GloVe 50d — reserved for future use)
├── lexicons/
│   ├── nrc-emolex.json     (NRC EmoLex, 14K terms, 8 Plutchik emotions + pos/neg)
│   ├── nrc-intensity.json  (NRC Affect Intensity, 4 emotions with continuous 0–1 scores)
│   ├── nrc-vad.json        (NRC VAD v2.1, ~20K terms, valence/arousal/dominance 0–1)
│   ├── vader.json          (VADER, ~7.5K terms, compound score -1 to +1)
│   └── sentiwordnet.json   (SentiWordNet 3.0, ~117K synsets, pos/neg/obj summing to 1)
└── wordnet/
    ├── lemmatizer.json     (WordNet exception lists + morphological rules)
    ├── pos-lookup.json     (word → most frequent POS, ~40K entries)
    └── synsets.json        (trimmed synset graph, depth=1, vocab-filtered, 12.9MB)
```

---

## Setup (for self-hosting or development)

### Prerequisites

- Python 3.8+ with `nltk` installed
- A web browser (Chrome, Firefox, Safari)
- ~500MB disk space for raw source files (temporary, during conversion)

### Step 1: Clone this repository

```bash
git clone https://github.com/EPSidebox/texturas.git
cd texturas
```

### Step 2: Download source data

Download the raw lexicon and vector files from their original sources. These are not included in the repository due to licensing.

| Resource | Download from | File you need |
|---|---|---|
| GloVe vectors | [nlp.stanford.edu/data/glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip) | `glove.6B.50d.txt` |
| NRC EmoLex | [saifmohammad.com](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) | `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` |
| NRC Affect Intensity | [saifmohammad.com](https://saifmohammad.com/WebPages/AffectIntensity.htm) | `NRC-Affect-Intensity-Lexicon.txt` |
| NRC VAD | [saifmohammad.com](https://saifmohammad.com/WebPages/nrc-vad.html) | `NRC-VAD-Lexicon.txt` |
| VADER | [github.com/cjhutto/vaderSentiment](https://github.com/cjhutto/vaderSentiment) | `vader_lexicon.txt` |
| SentiWordNet | [github.com/aesuli/SentiWordNet](https://github.com/aesuli/SentiWordNet) | `SentiWordNet_3.0.0.txt` |
| WordNet | via NLTK (see below) | Extracted automatically by the conversion script |

Place all downloaded files in a temporary folder (e.g., `raw/`).

### Step 3: Install Python dependencies

```bash
pip install nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 4: Run the conversion script

```bash
# Convert GloVe vectors (top 50,000 words, 50 dimensions)
python convert.py glove --input raw/glove.6B.50d.txt --output assets/vectors/

# Extract WordNet data
python convert.py wordnet --output assets/wordnet/

# Convert all sentiment lexicons
python convert.py lexicons \
  --nrc-emolex raw/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt \
  --nrc-intensity raw/NRC-Affect-Intensity-Lexicon.txt \
  --nrc-vad raw/NRC-VAD-Lexicon.txt \
  --vader raw/vader_lexicon.txt \
  --sentiwordnet raw/SentiWordNet_3.0.0.txt \
  --output assets/lexicons/
```

You can delete the `raw/` folder after conversion.

### Step 5: Configure and deploy

In `vellum/app.jsx`, set the `ASSET_BASE_URL` constant:

```javascript
var ASSET_BASE_URL = "https://YOUR_USERNAME.github.io/texturas/assets/";
```

Push to GitHub and enable GitHub Pages (Settings → Pages → Source: main branch).

---

## Usage

1. Open the tool in your browser
2. Go to **Input** — paste text, upload `.txt` files, or import `.xml` (TEI)
3. Use `---DOC: Label---` separators to paste multiple documents at once
4. Click **Analyze**
5. **Vellum tab** — explore the 3D grid; switch between Channels and Emotions pages; use the word panel to filter cells
6. **Weave tab** — read annotated text with six visual layers; toggle layers independently; use the word panel to highlight terms
7. **Output tab** — export TEI XML, CSV, or Markdown reports with configurable layer selection
8. **About tab** — citations, methodology, ethics note

### Shared Controls (right toolbar)

Both Vellum and Weave tabs share these parameters:
- **Arousal** — toggle arousal encoding (height in Vellum, underline thickness in Weave)
- **Flow** — spreading activation direction (Bi / ↑ hypernyms / ↓ hyponyms)
- **N** — top-N words for analysis (10–50)
- **Decay** — activation decay factor (0.30–0.80)
- **Grid** — grid resolution (10² / 20² / 30²)

---

## Lexicon Citations

If you use Textures in published work, please cite the relevant resources:

- **NRC EmoLex:** Mohammad, S.M. & Turney, P.D. (2013). Crowdsourcing a Word-Emotion Association Lexicon. *Computational Intelligence*, 29(3), 436–465.
- **NRC Affect Intensity:** Mohammad, S.M. (2018). Word Affect Intensities. In *Proceedings of LREC-2018*.
- **NRC VAD:** Mohammad, S.M. (2018). Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words. In *Proceedings of ACL-2018*.
- **VADER:** Hutto, C.J. & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. In *Proceedings of ICWSM-2014*.
- **SentiWordNet:** Baccianella, S., Esuli, A. & Sebastiani, F. (2010). SentiWordNet 3.0: An Enhanced Lexical Resource for Sentiment Analysis and Opinion Mining. In *Proceedings of LREC-2010*.
- **WordNet:** Princeton University. (2010). About WordNet. WordNet, Princeton University.
- **GloVe:** Pennington, J., Socher, R. & Manning, C.D. (2014). GloVe: Global Vectors for Word Representation. In *Proceedings of EMNLP-2014*.

### Methodology

- **Multi-lexicon approach:** Mitigates single-source bias per Czarnek & Stillwell (2022).
- **Spreading activation:** Adapted from Collins & Loftus (1975) for corpus relevance scoring.
- **Community detection:** Louvain modularity, Blondel et al. (2008).

### Ethics

Automated sentiment and emotion analysis produces preliminary indicators, not ground truth. Results should be interpreted in context and validated against close reading. See: Mohammad, S.M. (2022). Ethics Sheet for Automatic Emotion Recognition and Sentiment Analysis. *Computational Linguistics*, 48(2), 239–278.

---

## License

NRC lexicons are licensed for non-commercial research and education use only. VADER is MIT licensed. SentiWordNet is CC BY-SA 4.0. WordNet uses a BSD-like license. GloVe vectors are public domain. The Textures application code license is TBD.

---

*Textures is an open-source research tool. Static deployment. No data leaves the browser.*
