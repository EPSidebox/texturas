# ⬡ Texturas

**Multi-layered correlated textual analysis**

Texturas is a browser-based tool for identifying thematic, discursive, and affective patterns in text data. It layers corpus linguistics, distributional semantics, multi-lexicon sentiment analysis, and network modularity onto the same text — and cross-correlates these layers to surface patterns that no single method would reveal in isolation.

From Latin *textura* (weaving) — the root of "text." Plural: the many woven layers.

**Author:** Ernesto Peña · Northeastern University

---

## Features

- **Frequency analysis** — unigrams, bigrams, trigrams with stop word filtering (negation-aware)
- **Multi-lexicon sentiment** — NRC EmoLex (Plutchik 8 emotions), NRC Affect Intensity, NRC VAD (valence/arousal/dominance), VADER (polarity), SentiWordNet (sense-level pos/neg/obj)
- **Semantic expansion** — GloVe distributional similarity + WordNet taxonomic relationships (hypernyms, hyponyms, meronyms)
- **Co-occurrence networks** — windowed co-occurrence matrix with Louvain community detection
- **Cross-layer correlations** — community emotion profiles, sentiment trajectory, frequency × sentiment scatter, emotion co-occurrence
- **Multi-document support** — per-document and aggregate analysis with cross-document comparison
- **3D network visualization** — Three.js force-directed and seed-centric layouts with configurable visual channels
- **TEI XML export** — inline and standoff annotation models, per-document and corpus-level (`<teiCorpus>`)
- **CSV export** — per-token data with all annotation layers as columns
- **NLP pipeline** — POS tagging (hybrid lookup + suffix rules), WordNet lemmatization, sense-aware sentiment scoring

## How It Works

Texturas runs entirely in the browser. No server, no API keys. Static data assets (word vectors, sentiment lexicons, WordNet data) are loaded once and cached locally for instant subsequent visits.

### Analysis Pipeline

```
Text input
  → Tokenization (negation-preserving)
  → POS tagging (lookup table + suffix rules)
  → Lemmatization (WordNet morphological rules)
  → Frequency analysis (unigrams, bigrams, trigrams)
  → Multi-lexicon sentiment scoring (5 lexicons, sense-aware)
  → Semantic expansion (GloVe cosine similarity + WordNet synsets)
  → Co-occurrence matrix + community detection (Louvain)
  → Cross-layer correlation computation
  → TEI/CSV/report export
```

---

## Setup

### Prerequisites

- Python 3.8+ with `nltk` installed
- A web browser (Chrome, Firefox, Safari)
- ~500MB disk space for raw source files (temporary, during conversion)

### Step 1: Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/texturas.git
cd texturas
```

### Step 2: Download source data

You need to download the raw lexicon and vector files from their original sources. These are not included in the repository due to licensing.

| Resource | Download from | File you need |
|---|---|---|
| GloVe vectors | [nlp.stanford.edu/data/glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip) | `glove.6B.50d.txt` (from inside the zip) |
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

# Extract WordNet data (lemmatizer, POS lookup, synset relationships)
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

After conversion, your `assets/` folder should contain:

```
assets/
├── vectors/
│   ├── vocab.json      (~2 MB)
│   └── vectors.bin     (~10 MB)
├── lexicons/
│   ├── nrc-emolex.json
│   ├── nrc-intensity.json
│   ├── nrc-vad.json
│   ├── vader.json
│   └── sentiwordnet.json
└── wordnet/
    ├── lemmatizer.json
    ├── pos-lookup.json
    └── synsets.json
```

You can delete the `raw/` folder after conversion.

### Step 5: Configure the tool

Open `index.html` and set the `ASSET_BASE_URL` constant near the top:

```javascript
const ASSET_BASE_URL = "https://YOUR_USERNAME.github.io/texturas/assets/";
```

### Step 6: Deploy to GitHub Pages

Push everything to your repository, then enable GitHub Pages:

1. Go to your repository on github.com
2. Click **Settings** (tab at the top)
3. Scroll down to **Pages** (in the left sidebar)
4. Under **Source**, select **main** branch
5. Click **Save**

Your tool will be live at `https://YOUR_USERNAME.github.io/texturas/` within a few minutes.

---

## Usage

1. Open the tool in your browser
2. Go to the **Assets** tab to verify all resources loaded (you should see ✓ next to each)
3. Go to **Input** — paste text or upload `.txt` files
4. Click **Analyze** in the sidebar
5. Navigate the tabs: Frequencies → Sentiment → Correlations → Semantic → Co-occurrence → Network
6. Go to **Export** to download TEI XML, CSV, or summary reports

### Multi-document analysis

- Add documents with the **+ Add** button or **Upload .txt**
- Paste text with `---DOC: Label---` separators to auto-split
- After analysis, use the document selector bar to switch between individual documents and the **Aggregate** view

### 3D Network

- **Force mode**: co-occurrence drives proximity. Communities emerge spatially.
- **Seed mode**: click any node to anchor it at center. Connected terms orbit by co-occurrence strength.
- Drag to orbit, scroll to zoom, hover for details.

---

## Lexicon Citations

If you use Texturas in published work, please cite the tool and the relevant resources:

- **NRC EmoLex:** Mohammad, S.M. & Turney, P.D. (2013). Crowdsourcing a Word-Emotion Association Lexicon. *Computational Intelligence*, 29(3), 436–465.
- **NRC Affect Intensity:** Mohammad, S.M. (2018). Word Affect Intensities. In *Proceedings of LREC-2018*, Miyazaki, Japan.
- **NRC VAD:** Mohammad, S.M. (2018). Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words. In *Proceedings of ACL-2018*, Melbourne, Australia.
- **VADER:** Hutto, C.J. & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. In *Proceedings of ICWSM-2014*.
- **SentiWordNet:** Baccianella, S., Esuli, A. & Sebastiani, F. (2010). SentiWordNet 3.0: An Enhanced Lexical Resource for Sentiment Analysis and Opinion Mining. In *Proceedings of LREC-2010*, Valletta, Malta.
- **WordNet:** Princeton University. (2010). About WordNet. WordNet, Princeton University.
- **GloVe:** Pennington, J., Socher, R. & Manning, C.D. (2014). GloVe: Global Vectors for Word Representation. In *Proceedings of EMNLP-2014*.

For ethical considerations regarding automatic emotion detection, see:
- Mohammad, S.M. (2022). Ethics Sheet for Automatic Emotion Recognition and Sentiment Analysis. *Computational Linguistics*, 48(2), 239–278.

---

## Licensing

| Component | License |
|---|---|
| NRC Lexicons (EmoLex, Intensity, VAD) | Non-commercial research and educational use |
| VADER | MIT License |
| SentiWordNet 3.0 | CC BY-SA 4.0 |
| Princeton WordNet | WordNet License (BSD-like) |
| GloVe | Public Domain |
| Texturas (this tool) | TBD |

---

## Technical Stack

- **React** — UI framework
- **D3.js** — 2D visualizations (heatmaps, charts, trajectories)
- **Three.js** — 3D network visualization
- **Lodash** — Data manipulation
- No server, no API, no external services at runtime
