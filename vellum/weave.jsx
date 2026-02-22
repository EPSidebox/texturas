const { useState, useRef, useEffect, useCallback, useMemo } = React;

const ASSET_BASE_URL = "https://epsidebox.github.io/texturas/assets/";

const STOP_WORDS = new Set([
  "the","a","an","and","or","but","in","on","at","to","for","of","with","by","from",
  "is","are","was","were","be","been","being","have","has","had","do","does","did",
  "will","would","shall","should","may","might","can","could","this","that","these",
  "those","it","its","i","me","my","mine","we","us","our","ours","you","your","yours",
  "he","him","his","she","her","hers","they","them","their","theirs","what","which",
  "who","whom","whose","where","when","how","why","if","then","else","so","because",
  "as","than","too","very","just","about","above","after","again","all","also","am",
  "any","before","below","between","both","during","each","few","further","get","got",
  "here","into","more","most","much","only","other","out","over","own","same","some",
  "such","there","through","under","until","up","while","down","like","make","many",
  "now","off","once","one","onto","per","since","still","take","thing","things","think",
  "two","upon","well","went","yeah","yes","oh","um","uh","ah","okay","ok","really",
  "actually","basically","literally","right","know","going","go","come","say","said",
  "tell","told","see","look","want","need","way","back","even","around","something",
  "anything","everything","someone","anyone","everyone","always","sometimes","often",
  "already","yet","kind","sort","lot","bit","little","big","good","great","long","new",
  "old","first","last","next","time","times","day","days","year","years","people",
  "person","part","place","point","work","life","world","hand","week","end","case",
  "fact","area","number","given","able","made","set","used","using","another","different",
  "example","enough","however","important","include","including","keep","large","whether",
  "without","within","along","become","across","among","toward","towards","though",
  "although","either","rather","several","certain","less","likely","began","begun",
  "brought","thus","therefore","hence","moreover","furthermore","nevertheless",
  "nonetheless","meanwhile","accordingly","consequently","subsequently"
]);

const NEGATION = new Set([
  "not","no","never","neither","nor","don't","doesn't","didn't","won't","wouldn't",
  "can't","couldn't","shouldn't","isn't","aren't","wasn't","weren't","haven't","hasn't",
  "hadn't","cannot","nothing","nobody","none","nowhere"
]);

const EMOTIONS = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"];

const EC = {
  anger:"#ff6b6b", anticipation:"#f7dc6f", disgust:"#bb8fce", fear:"#85c1e9",
  joy:"#82e0aa", sadness:"#45b7d1", surprise:"#f0b27a", trust:"#4ecdc4"
};

const CC = ["#ff6b6b","#4ecdc4","#45b7d1","#f7dc6f","#bb8fce","#82e0aa","#f0b27a","#85c1e9","#f1948a","#73c6b6"];

const LAYER_CFG = [
  { id:"polarity", label:"Polarity", color:"#82e0aa", desc:"VADER text color" },
  { id:"emotion",  label:"Emotion",  color:"#f0b27a", desc:"Plutchik underline" },
  { id:"frequency",label:"Frequency",color:"#bb8fce", desc:"Corpus freq brightness" },
  { id:"relevance",label:"Relevance",color:"#4ecdc4", desc:"Activation font weight" },
  { id:"community",label:"Community",color:"#45b7d1", desc:"Louvain background" }
];

const EMO_LAYOUT = [
  { r:0,c:0,emo:"anticipation" },{ r:0,c:1,emo:"joy" },{ r:0,c:2,emo:"trust" },
  { r:1,c:0,emo:"anger" },{ r:1,c:1,emo:null },{ r:1,c:2,emo:"fear" },
  { r:2,c:0,emo:"disgust" },{ r:2,c:1,emo:"sadness" },{ r:2,c:2,emo:"surprise" }
];

const SFX = [
  ["tion","n"],["sion","n"],["ment","n"],["ness","n"],["ity","n"],["ance","n"],
  ["ence","n"],["ism","n"],["ist","n"],["ting","v"],["sing","v"],["ning","v"],
  ["ing","v"],["ated","v"],["ized","v"],["ed","v"],["ize","v"],["ify","v"],
  ["ously","r"],["ively","r"],["fully","r"],["ally","r"],["ly","r"],
  ["ful","a"],["ous","a"],["ive","a"],["able","a"],["ible","a"],["ical","a"],
  ["less","a"],["al","a"]
];

function mkPOS() {
  const o = { lk: null, ready: false };
  o.load = function(d) { o.lk = d; o.ready = true; };
  o.tag = function(w) {
    const l = w.toLowerCase();
    if (o.lk && o.lk[l]) return o.lk[l];
    for (let i = 0; i < SFX.length; i++) { if (l.endsWith(SFX[i][0])) return SFX[i][1]; }
    return "n";
  };
  return o;
}

function mkLem() {
  const o = { exc: null, rl: null, ready: false };
  o.load = function(d) { o.exc = d.exceptions; o.rl = d.rules; o.ready = true; };
  o.lemmatize = function(w, pos) {
    const l = w.toLowerCase();
    if (o.exc && o.exc[pos] && o.exc[pos][l]) return o.exc[pos][l];
    const rules = (o.rl && o.rl[pos]) || [];
    for (let i = 0; i < rules.length; i++) {
      if (l.endsWith(rules[i][0]) && l.length > rules[i][0].length)
        return l.slice(0, -rules[i][0].length) + rules[i][1];
    }
    return l;
  };
  return o;
}

function mkSyn() {
  const o = { d: null, ready: false };
  o.load = function(d) { o.d = d; o.ready = true; };
  o.getRels = function(w, p) {
    const e = o.d ? o.d[w + "#" + p] : null;
    if (!e) return { h: [], y: [], m: [] };
    return { h: e.hypernyms || [], y: e.hyponyms || [], m: e.meronyms || [] };
  };
  o.directed = function(w, p, flow) {
    const r = o.getRels(w, p);
    if (flow === "up") return Array.from(new Set([].concat(r.h, r.m)));
    if (flow === "down") return Array.from(new Set([].concat(r.y, r.m)));
    return Array.from(new Set([].concat(r.h, r.y, r.m)));
  };
  return o;
}

function mkSent() {
  const o = { el: null, int: null, vad: null, vdr: null, swn: null, ready: false };
  function ck() { o.ready = !!(o.el || o.int || o.vad || o.vdr || o.swn); }
  o.lEl = function(d) { o.el = d; ck(); };
  o.lInt = function(d) { o.int = d; ck(); };
  o.lVad = function(d) { o.vad = d; ck(); };
  o.lVdr = function(d) { o.vdr = d; ck(); };
  o.lSwn = function(d) { o.swn = d; ck(); };
  o.score = function(lem, pos) {
    const r = { lemma: lem, pos: pos };
    if (o.el && o.el[lem]) r.emolex = o.el[lem];
    if (o.int && o.int[lem]) r.intensity = o.int[lem];
    if (o.vad && o.vad[lem]) r.vad = o.vad[lem];
    if (o.vdr && o.vdr[lem] !== undefined) r.vader = o.vdr[lem];
    const k = lem + "#" + pos;
    if (o.swn && o.swn[k]) { r.swn = o.swn[k]; }
    else {
      const tries = ["n","v","a","r"];
      for (let i = 0; i < tries.length; i++) {
        const v = o.swn ? o.swn[lem + "#" + tries[i]] : null;
        if (v) { r.swn = v; break; }
      }
    }
    r.has = !!(r.emolex || r.intensity || r.vad || r.vader !== undefined || r.swn);
    return r;
  };
  return o;
}

function spreadAct(fMap, syn, pos, depth, decay, flow) {
  if (!syn.ready) return Object.assign({}, fMap);
  const corpus = new Set(Object.keys(fMap));
  const scores = {};
  for (const l of corpus) scores[l] = fMap[l] || 0;
  for (const [l, f] of Object.entries(fMap)) {
    if (f === 0) continue;
    const p = pos.ready ? pos.tag(l) : "n";
    let front = [{ w: l, p: p }];
    const vis = new Set([l]);
    for (let h = 1; h <= depth; h++) {
      const nf = [];
      for (const nd of front) {
        const tgts = syn.directed(nd.w, nd.p, flow);
        for (const t of tgts) {
          if (vis.has(t)) continue;
          vis.add(t);
          const amt = f * Math.pow(decay, h);
          if (corpus.has(t)) scores[t] = (scores[t] || 0) + amt;
          nf.push({ w: t, p: pos.ready ? pos.tag(t) : "n" });
        }
      }
      front = nf;
    }
  }
  return scores;
}

function getNg(ts, n) {
  const g = [];
  for (let i = 0; i <= ts.length - n; i++) g.push(ts.slice(i, i + n).join(" "));
  return _.chain(g).countBy().toPairs().sortBy(1).reverse().value();
}

function analyzeForWeave(text, eng, topN, winSize, wnDepth, decay, flow) {
  const norm = text.replace(/[\u2018\u2019\u201A\u201B]/g, "'").replace(/[\u201C\u201D\u201E\u201F]/g, '"');
  const paras = norm.split(/\n\s*\n+/).filter(function(p) { return p.trim(); });
  const allWords = [];
  const renderParas = paras.map(function(para) {
    const parts = [];
    const rx = /([\w'-]+)|(\s+)|([^\w\s'-]+)/g;
    let m;
    while ((m = rx.exec(para)) !== null) {
      if (m[1]) {
        const surface = m[1], lower = surface.toLowerCase();
        const isStop = STOP_WORDS.has(lower) && !NEGATION.has(lower);
        const p = eng.pos.ready ? eng.pos.tag(lower) : "n";
        const lemma = eng.lem.ready ? eng.lem.lemmatize(lower, p) : lower;
        if (!isStop) allWords.push(lemma);
        parts.push({ type: "word", surface, lower, lemma, pos: p, isStop });
      } else {
        parts.push({ type: "other", surface: m[0] });
      }
    }
    return parts;
  });

  const freqMap = _.countBy(allWords);
  const freqPairs = _.chain(freqMap).toPairs().sortBy(1).reverse().value();
  const topWords = freqPairs.slice(0, topN).map(function(f) { return f[0]; });
  const maxFreq = freqPairs[0] ? freqPairs[0][1] : 1;
  const ng2 = getNg(allWords, 2).slice(0, topN);
  const ng3 = getNg(allWords, 3).slice(0, topN);

  const ws = new Set(topWords);
  const mx = {};
  topWords.forEach(function(a) { mx[a] = {}; topWords.forEach(function(b) { mx[a][b] = 0; }); });
  for (let i = 0; i < allWords.length; i++) {
    if (!ws.has(allWords[i])) continue;
    for (let j = Math.max(0, i - winSize); j <= Math.min(allWords.length - 1, i + winSize); j++) {
      if (i !== j && ws.has(allWords[j])) mx[allWords[i]][allWords[j]]++;
    }
  }

  const n = topWords.length;
  let cm = topWords.map(function(_, i) { return i; });
  let tw2 = 0;
  const wt = {};
  topWords.forEach(function(a) { wt[a] = 0; topWords.forEach(function(b) { const w = mx[a][b] || 0; tw2 += w; wt[a] += w; }); });
  tw2 /= 2;

  if (tw2 > 0) {
    let imp = true, it = 0;
    while (imp && it < 20) {
      imp = false; it++;
      for (let i = 0; i < n; i++) {
        let bc = cm[i], bg = 0;
        const ucs = Array.from(new Set(cm));
        for (const c of ucs) {
          if (c === cm[i]) continue;
          let g = 0;
          for (let j = 0; j < n; j++) {
            if (cm[j] !== c) continue;
            g += (mx[topWords[i]][topWords[j]] || 0) - (wt[topWords[i]] * wt[topWords[j]]) / (2 * tw2);
          }
          if (g > bg) { bg = g; bc = c; }
        }
        if (bc !== cm[i]) { cm[i] = bc; imp = true; }
      }
    }
  }

  const uComms = Array.from(new Set(cm));
  const commMap = {};
  topWords.forEach(function(w, i) { commMap[w] = uComms.indexOf(cm[i]); });
  const relevanceMap = spreadAct(freqMap, eng.syn, eng.pos, wnDepth, decay, flow);
  const maxRel = Math.max.apply(null, Object.values(relevanceMap).concat([1]));
  const sentCache = {};
  function getSent(lem, p) {
    if (sentCache[lem]) return sentCache[lem];
    const s = eng.sent.ready ? eng.sent.score(lem, p) : {};
    sentCache[lem] = s; return s;
  }
  const enriched = renderParas.map(function(parts) {
    return parts.map(function(t) {
      if (t.type !== "word") return t;
      const s = getSent(t.lemma, t.pos);
      return Object.assign({}, t, {
        vader: s.vader != null ? s.vader : null,
        emotions: s.emolex ? EMOTIONS.filter(function(e) { return s.emolex[e]; }) : [],
        vad: s.vad || null, frequency: freqMap[t.lemma] || 0,
        relevance: relevanceMap[t.lemma] || 0,
        community: commMap[t.lemma] != null ? commMap[t.lemma] : null,
        isTopN: ws.has(t.lemma)
      });
    });
  });
  return { enriched, freqPairs, freqMap, relevanceMap, maxFreq, maxRel, topWords, commMap, ng2, ng3 };
}

const DB_NAME = "texturas-cache";
function openDB() {
  return new Promise(function(resolve, reject) {
    const q = indexedDB.open(DB_NAME, 1);
    q.onupgradeneeded = function() { q.result.createObjectStore("assets"); };
    q.onsuccess = function() { resolve(q.result); };
    q.onerror = function() { reject(q.error); };
  });
}
async function cGet(k) {
  try { const db = await openDB(); return new Promise(function(r) { const t = db.transaction("assets","readonly"); const q = t.objectStore("assets").get(k); q.onsuccess = function() { r(q.result || null); }; q.onerror = function() { r(null); }; }); } catch(e) { return null; }
}
async function cSet(k, v) {
  try { const db = await openDB(); return new Promise(function(r) { const t = db.transaction("assets","readwrite"); t.objectStore("assets").put(v, k); t.oncomplete = function() { r(true); }; t.onerror = function() { r(false); }; }); } catch(e) { return false; }
}
async function loadAsset(key, path, bin, cb) {
  const c = await cGet(key); if (c) return c;
  const b = ASSET_BASE_URL.endsWith("/") ? ASSET_BASE_URL : ASSET_BASE_URL + "/";
  if (cb) cb("Fetching " + path + "...");
  try { const r = await fetch(b + path); if (!r.ok) return null; const d = bin ? await r.arrayBuffer() : await r.json(); await cSet(key, d); return d; } catch(e) { return null; }
}

function EmoBars({ emotions, arousal, showEmo, showAro, enabledSlots }) {
  const wrap = { position:"absolute", bottom:0, left:0, right:0, height:8, pointerEvents:"none" };
  if (!showEmo) return <span style={wrap} />;
  const active = EMOTIONS.filter(function(e) { return enabledSlots.has(e); });
  const present = new Set(emotions);
  if (!active.length) return <span style={wrap} />;
  const h = showAro ? Math.max(2, Math.round((arousal != null ? arousal : 0.5) * 8)) : 6;
  return (
    <span style={Object.assign({}, wrap, { display:"flex", gap:0, alignItems:"flex-end", padding:"0 1px" })}>
      {active.map(function(e) {
        return <span key={e} style={{ flex:1, height: present.has(e) ? h : 0, background: present.has(e) ? EC[e] : "transparent", borderRadius:"1px 1px 0 0", opacity:0.85, minWidth:1, maxWidth:4 }} />;
      })}
    </span>
  );
}

function WeaveTooltip({ token, x, y }) {
  if (!token || token.isStop) return null;
  const t = token;
  return (
    <div style={{ position:"fixed", left: Math.min(x + 12, window.innerWidth - 280), top: Math.min(y - 10, window.innerHeight - 300), zIndex:1000, padding:"10px 14px", background:"#1a1a1aee", border:"1px solid #444", borderRadius:6, fontFamily:"monospace", fontSize:11, color:"#ccc", pointerEvents:"none", maxWidth:260, lineHeight:1.7, backdropFilter:"blur(4px)" }}>
      <div style={{ fontSize:13, color:"#4ecdc4", marginBottom:4, fontWeight:"bold" }}>{t.lemma} <span style={{ color:"#666", fontWeight:"normal" }}>({t.pos})</span></div>
      {t.vader !== null && <div>VADER: <span style={{ color: t.vader > 0.05 ? "#82e0aa" : t.vader < -0.05 ? "#ff6b6b" : "#888" }}>{t.vader > 0 ? "+" : ""}{t.vader.toFixed(3)}</span></div>}
      {t.emotions.length > 0 && <div>Emotions: {t.emotions.map(function(e) { return <span key={e} style={{ color: EC[e], marginRight:4 }}>{e}</span>; })}</div>}
      <div>Freq: <span style={{ color:"#bb8fce" }}>{t.frequency}</span> {"\u00B7"} Relev: <span style={{ color:"#4ecdc4" }}>{t.relevance.toFixed(1)}</span></div>
      {t.community !== null && <div>Community: <span style={{ color: CC[t.community % CC.length] }}>C{t.community + 1}</span></div>}
      {t.vad && <div>V={t.vad.v ? t.vad.v.toFixed(2) : "?"} A={t.vad.a ? t.vad.a.toFixed(2) : "?"} D={t.vad.d ? t.vad.d.toFixed(2) : "?"}</div>}
    </div>
  );
}

function WeaveMinimap({ enriched, layers, enabledSlots, maxFreq, maxRel, scrollFrac, viewFrac, onSeek, height }) {
  const cvRef = useRef();
  const flatWords = useMemo(function() {
    const out = [];
    enriched.forEach(function(para) { para.forEach(function(t) { if (t.type === "word") out.push(t); }); out.push(null); });
    return out;
  }, [enriched]);
  useEffect(function() {
    const cv = cvRef.current; if (!cv) return;
    const ctx = cv.getContext("2d"); const h = height || 400;
    cv.width = 80; cv.height = h; ctx.fillStyle = "#0d0d0d"; ctx.fillRect(0, 0, 80, h);
    const total = flatWords.length; if (!total) return;
    const rowH = Math.max(1, Math.min(3, h / Math.ceil(total / 16)));
    const cols = 16, cw = Math.floor(80 / cols);
    let row = 0, col = 0;
    flatWords.forEach(function(t) {
      if (!t) { row++; col = 0; return; }
      const yy = row * rowH, xx = col * cw; if (yy > h) return;
      let c = "#333";
      if (!t.isStop) {
        if (layers.polarity && t.vader !== null) c = t.vader > 0.05 ? "#82e0aa" : t.vader < -0.05 ? "#ff6b6b" : "#666";
        else if (layers.community && t.community !== null) c = CC[t.community % CC.length];
        else if (layers.emotion) { const fe = (t.emotions || []).filter(function(e) { return enabledSlots.has(e); }); c = fe.length ? EC[fe[0]] : "#555"; }
        else if (layers.relevance) { const rn = maxRel > 0 ? Math.log(1 + t.relevance) / Math.log(1 + maxRel) : 0; const b = Math.round(50 + rn * 200); c = "rgb(" + b + "," + Math.round(b * 1.2) + "," + b + ")"; }
        else if (layers.frequency) { const fn = maxFreq > 0 ? Math.log(1 + t.frequency) / Math.log(1 + maxFreq) : 0; const b = Math.round(40 + fn * 180); c = "rgb(" + b + "," + b + "," + b + ")"; }
        else c = "#555";
      } else c = "#222";
      ctx.fillStyle = c; ctx.fillRect(xx + 1, yy, cw - 1, Math.max(1, rowH - 1));
      col++; if (col >= cols) { col = 0; row++; }
    });
    const mapH = (row + 1) * rowH;
    const vpY = scrollFrac * Math.min(mapH, h), vpH = Math.max(8, viewFrac * Math.min(mapH, h));
    ctx.fillStyle = "rgba(78,205,196,0.15)"; ctx.fillRect(0, vpY, 80, vpH);
    ctx.strokeStyle = "#4ecdc466"; ctx.lineWidth = 1; ctx.strokeRect(0.5, vpY + 0.5, 79, vpH - 1);
  }, [flatWords, layers, enabledSlots, maxFreq, maxRel, scrollFrac, viewFrac, height]);
  const handleClick = useCallback(function(e) { const cv = cvRef.current; if (!cv) return; const r = cv.getBoundingClientRect(); onSeek(Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))); }, [onSeek]);
  const dragRef = useRef(false);
  useEffect(function() { const up = function() { dragRef.current = false; }; window.addEventListener("mouseup", up); return function() { window.removeEventListener("mouseup", up); }; }, []);
  return <canvas ref={cvRef} style={{ width:80, flexShrink:0, cursor:"pointer", borderLeft:"1px solid #2a2a2a", borderRight:"1px solid #2a2a2a" }} onMouseDown={function(e) { dragRef.current = true; handleClick(e); }} onMouseMove={function(e) { if (dragRef.current) handleClick(e); }} />;
}

function WeaveReader({ enriched, layers, highlightLemma, maxFreq, maxRel, onHover, onClick, enabledSlots, showArousal, scrollRef, onScroll, gridSize }) {
  if (!enriched || !enriched.length) return <div style={{ color:"#555", textAlign:"center", marginTop:60, fontSize:13 }}>Run analysis to see annotated text.</div>;
  const gs = gridSize || 10, cells = gs * gs;
  let totalWords = 0;
  enriched.forEach(function(para) { para.forEach(function(t) { if (t.type === "word" && t.lower && t.lower.length > 1) totalWords++; }); });
  const base = Math.floor(totalWords / cells), extra = totalWords % cells;
  const binStarts = useMemo(function() {
    const s = new Set(); let pos = 0;
    for (let i = 0; i < cells; i++) { s.add(pos); pos += i < extra ? base + 1 : base; }
    return s;
  }, [totalWords, cells, base, extra]);
  function binLabel(wi) {
    let pos = 0;
    for (let i = 0; i < cells; i++) { const sz = i < extra ? base + 1 : base; if (wi >= pos && wi < pos + sz) return "[" + (Math.floor(i / gs) + 1) + "," + (i % gs + 1) + "]"; pos += sz; }
    return null;
  }
  let wordIdx = 0;
  return (
    <div ref={scrollRef} onScroll={onScroll} style={{ flex:1, overflowY:"auto", padding:"24px 32px", lineHeight:2.6, wordSpacing:"0.06em" }}>
      {enriched.map(function(para, pi) {
        return (
          <div key={pi} style={{ marginBottom:20 }}>
            {para.map(function(t, ti) {
              if (t.type !== "word") return <span key={ti} style={{ fontFamily:"monospace" }}>{t.surface}</span>;
              const isWord = t.lower && t.lower.length > 1;
              const wi = isWord ? wordIdx : -1;
              if (isWord) wordIdx++;
              const showMarker = isWord && binStarts.has(wi);
              const marker = showMarker ? binLabel(wi) : null;
              if (t.isStop) return (
                <span key={ti} style={{ fontFamily:"monospace" }}>
                  {showMarker && <span data-bin={wi} title={marker} style={{ display:"inline-block", width:0, height:"1.1em", borderLeft:"1.5px solid #4ecdc444", marginRight:2, verticalAlign:"middle" }} />}
                  <span style={{ color:"#444" }}>{t.surface}</span>
                </span>
              );
              const s = { fontFamily:"monospace", cursor:"pointer", position:"relative", display:"inline-block", transition:"all 0.15s", padding:"1px 0px", paddingBottom:10, borderRadius:"2px", fontWeight:400 };
              s.color = (layers.polarity && t.vader !== null) ? (t.vader > 0.05 ? "#82e0aa" : t.vader < -0.05 ? "#ff6b6b" : "#999") : "#ccc";
              if (layers.frequency) { const fn = maxFreq > 0 ? Math.log(1 + t.frequency) / Math.log(1 + maxFreq) : 0; s.opacity = 0.25 + fn * 0.75; }
              if (layers.relevance) { const rn = maxRel > 0 ? Math.log(1 + t.relevance) / Math.log(1 + maxRel) : 0; s.fontWeight = Math.round(100 + rn * 500); }
              if (layers.community && t.community !== null) s.backgroundColor = CC[t.community % CC.length] + "1a";
              if (highlightLemma && t.lemma === highlightLemma) { s.outline = "2px solid #4ecdc4"; s.outlineOffset = "3px"; s.borderRadius = "3px"; s.textShadow = "0 0 8px #4ecdc466"; if (!s.backgroundColor) s.backgroundColor = "#4ecdc40d"; }
              const filtEmo = layers.emotion ? t.emotions.filter(function(e) { return enabledSlots.has(e); }) : [];
              return (
                <span key={ti} style={{ fontFamily:"monospace" }}>
                  {showMarker && <span data-bin={wi} title={marker} style={{ display:"inline-block", width:0, height:"1.1em", borderLeft:"1.5px solid #4ecdc466", marginRight:2, verticalAlign:"middle" }} />}
                  <span style={s} onMouseEnter={function(e) { onHover(t, e.clientX, e.clientY); }} onMouseMove={function(e) { onHover(t, e.clientX, e.clientY); }} onMouseLeave={function() { onHover(null, 0, 0); }} onClick={function() { onClick(t.lemma); }}>
                    {t.surface}
                    <EmoBars emotions={filtEmo} arousal={t.vad ? t.vad.a : null} showEmo={layers.emotion} showAro={showArousal} enabledSlots={enabledSlots} />
                  </span>
                </span>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

function WeaveWordPanel({ result, topN, highlightLemma, onClickWord, ngMode, setNgMode, sortBy, setSortBy }) {
  const isUni = ngMode === "1";
  const raw = isUni ? result.freqPairs.slice(0, topN) : ngMode === "2" ? result.ng2.slice(0, topN) : result.ng3.slice(0, topN);
  const sorted = useMemo(function() {
    if (!isUni || sortBy !== "relevance") return raw;
    return raw.slice().sort(function(a, b) { return (result.relevanceMap[b[0]] || 0) - (result.relevanceMap[a[0]] || 0); });
  }, [raw, isUni, sortBy, result.relevanceMap]);
  const maxF = raw[0] ? raw[0][1] : 1;
  const maxR = isUni ? Math.max.apply(null, raw.map(function(x) { return result.relevanceMap[x[0]] || 0; }).concat([1])) : 1;
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:1, minWidth:0, height:"100%" }}>
      <button onClick={function() { onClickWord(null); }} style={{ padding:"4px 8px", marginBottom:2, background: !highlightLemma ? "#4ecdc4" : "#1a1a1a", color: !highlightLemma ? "#111" : "#888", border: "1px solid " + (!highlightLemma ? "#4ecdc4" : "#333"), borderRadius:4, cursor:"pointer", fontSize:11, fontFamily:"monospace", fontWeight: !highlightLemma ? "bold" : "normal", textAlign:"left" }}>All</button>
      <div style={{ display:"flex", gap:0, marginBottom:3, border:"1px solid #333", borderRadius:3, overflow:"hidden" }}>
        {[["1","1"],["2","2"],["3","3"]].map(function(pair) {
          return <button key={pair[0]} onClick={function() { setNgMode(pair[0]); onClickWord(null); }} style={{ flex:1, padding:"3px 0", background: ngMode === pair[0] ? "#bb8fce" : "#1a1a1a", color: ngMode === pair[0] ? "#111" : "#666", border:"none", cursor:"pointer", fontSize:10, fontFamily:"monospace", fontWeight: ngMode === pair[0] ? "bold" : "normal" }}>{pair[1]}</button>;
        })}
      </div>
      {isUni && (
        <div style={{ display:"flex", gap:0, marginBottom:3, border:"1px solid #333", borderRadius:3, overflow:"hidden" }}>
          {[["freq","Freq"],["relevance","Relev"]].map(function(pair) {
            return <button key={pair[0]} onClick={function() { setSortBy(pair[0]); }} style={{ flex:1, padding:"3px 0", background: sortBy === pair[0] ? "#45b7d1" : "#1a1a1a", color: sortBy === pair[0] ? "#111" : "#555", border:"none", cursor:"pointer", fontSize:9, fontFamily:"monospace" }}>{pair[1]}</button>;
          })}
        </div>
      )}
      <div style={{ flex:1, overflowY:"auto", display:"flex", flexDirection:"column", gap:1 }}>
        {sorted.map(function(pair) {
          const w = pair[0], c = pair[1], rel = isUni ? (result.relevanceMap[w] || 0) : 0, isHL = highlightLemma === w;
          return (
            <button key={w} onClick={function() { onClickWord(w === highlightLemma ? null : w); }} style={{ padding:"3px 8px", background: isHL ? "#1a2a2a" : "#111", color: isHL ? "#ccc" : "#aaa", border: "1px solid " + (isHL ? "#4ecdc444" : "#1a1a1a"), borderRadius:3, cursor:"pointer", fontSize:10, fontFamily:"monospace", textAlign:"left", display:"flex", alignItems:"center", gap:6 }}>
              <span style={{ flex:1, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", color: isHL ? "#4ecdc4" : "inherit" }}>{w}</span>
              <div style={{ width:46, display:"flex", flexDirection:"column", gap:1, flexShrink:0 }}>
                <div style={{ height:3, background:"#222", borderRadius:1, overflow:"hidden" }}><div style={{ width: (c / maxF * 100) + "%", height:"100%", background:"#4ecdc4", borderRadius:1, opacity: isHL ? 0.9 : 0.5 }} /></div>
                {isUni && maxR > 0 && <div style={{ height:3, background:"#222", borderRadius:1, overflow:"hidden" }}><div style={{ width: (rel / maxR * 100) + "%", height:"100%", background:"#45b7d1", borderRadius:1, opacity: isHL ? 0.9 : 0.5 }} /></div>}
              </div>
              <span style={{ width:24, textAlign:"right", fontSize:9, color: isHL ? "#888" : "#555", flexShrink:0 }}>{c}</span>
            </button>
          );
        })}
      </div>
      <div style={{ display:"flex", gap:8, marginTop:4, fontSize:9, color:"#888" }}>
        <span><span style={{ color:"#4ecdc4" }}>{"\u2014"}</span> freq</span>
        {isUni && <span><span style={{ color:"#45b7d1" }}>{"\u2014"}</span> relev</span>}
      </div>
    </div>
  );
}

function EmoToggle({ enabledSlots, setEnabledSlots }) {
  return (
    <div style={{ display:"inline-grid", gridTemplateColumns:"repeat(3,10px)", gap:1 }}>
      {EMO_LAYOUT.map(function(item, i) {
        const key = item.emo || "center", on = enabledSlots.has(key), col = item.emo ? EC[item.emo] : "#ffffff";
        return <div key={i} onClick={function() { setEnabledSlots(function(prev) { const n = new Set(prev); if (n.has(key)) n.delete(key); else n.add(key); return n; }); }} title={item.emo || "relevance"} style={{ width:10, height:10, borderRadius:1, background: on ? col : "#555", opacity: on ? 1 : 0.5, cursor:"pointer", border: "1px solid " + (on ? col + "66" : "#444") }} />;
      })}
    </div>
  );
}

function WeaveStandalone()
  const [docs, setDocs] = useState([{ id:"d1", label:"Document 1", text:"" }]);
  const [activeInputDoc, setActiveInputDoc] = useState("d1");
  const [tab, setTab] = useState("input");
  const [topN, setTopN] = useState(25);
  const [wnDepth, setWnDepth] = useState(2);
  const [decay, setDecay] = useState(0.5);
  const [flow, setFlow] = useState("bi");
  const [winSize] = useState(5);
  const [weavePerDoc, setWeavePerDoc] = useState({});
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");
  const [wLayers, setWLayers] = useState({ polarity:true, emotion:true, frequency:false, relevance:false, community:false });
  const [showArousal, setShowArousal] = useState(false);
  const [enabledSlots, setEnabledSlots] = useState(new Set(EMOTIONS.concat(["center"])));
  const [gridSize, setGridSize] = useState(10);
  const [wHighlight, setWHighlight] = useState(null);
  const [wHovTok, setWHovTok] = useState(null);
  const [wHovPos, setWHovPos] = useState({ x:0, y:0 });
  const [wNgMode, setWNgMode] = useState("1");
  const [wSortBy, setWSortBy] = useState("freq");
  const [wActiveDoc, setWActiveDoc] = useState(null);
  const [showParams, setShowParams] = useState(false);
  const readerRef = useRef();
  const [scrollFrac, setScrollFrac] = useState(0);
  const [viewFrac, setViewFrac] = useState(1);
  const [contentH, setContentH] = useState(400);

  const handleReaderScroll = useCallback(function() {
    const el = readerRef.current; if (!el) return;
    setScrollFrac(el.scrollHeight > el.clientHeight ? el.scrollTop / (el.scrollHeight - el.clientHeight) : 0);
    setViewFrac(el.scrollHeight > 0 ? el.clientHeight / el.scrollHeight : 1);
  }, []);
  const handleMinimapSeek = useCallback(function(frac) {
    const el = readerRef.current; if (!el) return;
    el.scrollTop = frac * (el.scrollHeight - el.clientHeight);
  }, []);
  const contentRef = useRef();
  useEffect(function() {
    if (!contentRef.current) return;
    const ro = new ResizeObserver(function(en) { setContentH(en[0].contentRect.height); });
    ro.observe(contentRef.current); return function() { ro.disconnect(); };
  }, []);

  const eng = useRef({ pos: mkPOS(), lem: mkLem(), syn: mkSyn(), sent: mkSent() });
  useEffect(function() {
    let cancelled = false;
    (async function() {
      const e = eng.current;
      const pd = await loadAsset("w-p", "wordnet/pos-lookup.json", false, setMsg); if (pd && !cancelled) e.pos.load(pd);
      const ld = await loadAsset("w-l", "wordnet/lemmatizer.json", false, setMsg); if (ld && !cancelled) e.lem.load(ld);
      const sd = await loadAsset("w-s", "wordnet/synsets.json", false, setMsg); if (sd && !cancelled) e.syn.load(sd);
      const el = await loadAsset("l-e", "lexicons/nrc-emolex.json", false, setMsg); if (el && !cancelled) e.sent.lEl(el);
      const ni = await loadAsset("l-i", "lexicons/nrc-intensity.json", false, setMsg); if (ni && !cancelled) e.sent.lInt(ni);
      const nv = await loadAsset("l-v", "lexicons/nrc-vad.json", false, setMsg); if (nv && !cancelled) e.sent.lVad(nv);
      const va = await loadAsset("l-d", "lexicons/vader.json", false, setMsg); if (va && !cancelled) e.sent.lVdr(va);
      const sw = await loadAsset("l-s", "lexicons/sentiwordnet.json", false, setMsg); if (sw && !cancelled) e.sent.lSwn(sw);
      if (!cancelled) setMsg("");
    })();
    return function() { cancelled = true; };
  }, []);

  const validDocs = docs.filter(function(d) { return d.text.trim(); });
  const analyzedIds = Object.keys(weavePerDoc);

  const runAnalysis = useCallback(function() {
    if (!validDocs.length) return;
    setLoading(true); setMsg("Analyzing...");
    setTimeout(function() {
      const e = eng.current, wdr = {};
      validDocs.forEach(function(d) { wdr[d.id] = analyzeForWeave(d.text, e, topN, winSize, wnDepth, decay, flow); });
      setWeavePerDoc(wdr); setWActiveDoc(validDocs[0].id); setWHighlight(null);
      setLoading(false); setMsg(""); setTab("weave"); setTimeout(handleReaderScroll, 100);
    }, 50);
  }, [docs, topN, wnDepth, decay, flow, winSize]);

  function rerunFlow(v) { setFlow(v); if (!validDocs.length || !analyzedIds.length) return; setTimeout(function() { const e = eng.current, wdr = {}; validDocs.forEach(function(d) { wdr[d.id] = analyzeForWeave(d.text, e, topN, winSize, wnDepth, decay, v); }); setWeavePerDoc(wdr); }, 50); }
  const rerunTopN = useCallback(function(n) { setTopN(n); if (!validDocs.length || !analyzedIds.length) return; setTimeout(function() { const e = eng.current, wdr = {}; validDocs.forEach(function(d) { wdr[d.id] = analyzeForWeave(d.text, e, n, winSize, wnDepth, decay, flow); }); setWeavePerDoc(wdr); }, 50); }, [docs, wnDepth, decay, flow, winSize]);
  const rerunDecay = useCallback(function(d) { setDecay(d); if (!validDocs.length || !analyzedIds.length) return; setTimeout(function() { const e = eng.current, wdr = {}; validDocs.forEach(function(doc) { wdr[doc.id] = analyzeForWeave(doc.text, e, topN, winSize, wnDepth, d, flow); }); setWeavePerDoc(wdr); }, 50); }, [docs, topN, wnDepth, flow, winSize]);
  function toggleWLayer(id) { setWLayers(function(prev) { const next = Object.assign({}, prev); next[id] = !next[id]; if (Object.values(next).every(function(v) { return !v; })) return prev; return next; }); }

  const activeWR = wActiveDoc ? weavePerDoc[wActiveDoc] : null;
  const curDoc = docs.find(function(d) { return d.id === activeInputDoc; });
  const hasMarkers = curDoc && curDoc.text && curDoc.text.includes("---DOC");

  function addDoc() { const id = "d" + Date.now(); setDocs(function(d) { return d.concat([{ id: id, label: "Document " + (d.length + 1), text: "" }]); }); setActiveInputDoc(id); }
  function rmDoc(id) { if (docs.length <= 1) return; setDocs(function(d) { return d.filter(function(x) { return x.id !== id; }); }); if (activeInputDoc === id) setActiveInputDoc(docs[0].id); }
  function updDoc(id, f, v) { setDocs(function(d) { return d.map(function(x) { return x.id === id ? Object.assign({}, x, { [f]: v }) : x; }); }); }
  function handleFiles(files) { Array.from(files).forEach(function(f) { if (!f.name.endsWith(".txt")) return; const id = "d" + Date.now() + "_" + Math.random().toString(36).slice(2, 6); const reader = new FileReader(); reader.onload = function(ev) { setDocs(function(d) { return d.filter(function(x) { return x.text.trim(); }).concat([{ id: id, label: f.name.replace(".txt", ""), text: ev.target.result }]); }); }; reader.readAsText(f); }); }
  function parseSep(text) { const parts = text.split(/---DOC(?::?\s*([^-]*))?\s*---/i); const result = []; let pl = null; for (let i = 0; i < parts.length; i++) { const t = parts[i] ? parts[i].trim() : ""; if (!t) continue; if (i % 2 === 1) pl = t; else { result.push({ id: "d" + Date.now() + "_" + i, label: pl || "Document " + (result.length + 1), text: t }); pl = null; } } return result.length > 1 ? result : null; }

  const BtnTab = function(props) { return <button onClick={function() { setTab(props.id); }} style={{ padding:"10px 14px", background: tab === props.id ? "#1a1a1a" : "transparent", color: tab === props.id ? "#4ecdc4" : "#888", border:"none", borderBottom: tab === props.id ? "2px solid #4ecdc4" : "2px solid transparent", cursor:"pointer", fontSize:12, fontFamily:"monospace" }}>{props.label}</button>; };

  return (
    <div style={{ background:"#111", color:"#ddd", minHeight:"100vh", fontFamily:"monospace", display:"flex", flexDirection:"column" }}>
      <div style={{ padding:"12px 20px", borderBottom:"1px solid #2a2a2a", display:"flex", alignItems:"center", gap:12 }}>
        <span style={{ fontSize:18, color:"#4ecdc4", fontWeight:"bold" }}>{"\u2B21"} Texturas</span>
        <span style={{ fontSize:11, color:"#555" }}>Weave standalone</span>
        <div style={{ marginLeft:"auto", display:"flex", gap:8 }}>
          {eng.current.sent.ready && <span style={{ fontSize:10, color:"#f7dc6f" }}>{"\u25CF"} sent</span>}
          {eng.current.pos.ready && <span style={{ fontSize:10, color:"#bb8fce" }}>{"\u25CF"} nlp</span>}
          {eng.current.syn.ready && <span style={{ fontSize:10, color:"#45b7d1" }}>{"\u25CF"} wn</span>}
        </div>
      </div>

      <div style={{ display:"flex", borderBottom:"1px solid #2a2a2a" }}>
        <BtnTab id="input" label="Input" />
        <BtnTab id="weave" label="Weave" />
      </div>

      {tab === "weave" && analyzedIds.length > 0 && (
        <div style={{ display:"flex", gap:4, padding:"8px 20px", borderBottom:"1px solid #2a2a2a", background:"#151515" }}>
          {validDocs.filter(function(d) { return weavePerDoc[d.id]; }).map(function(d) {
            return <button key={d.id} onClick={function() { setWActiveDoc(d.id); setWHighlight(null); }} style={{ padding:"4px 12px", borderRadius:3, border: "1px solid " + (wActiveDoc === d.id ? "#45b7d1" : "#333"), background: wActiveDoc === d.id ? "#45b7d1" : "#1a1a1a", color: wActiveDoc === d.id ? "#111" : "#888", fontSize:11, fontFamily:"monospace", cursor:"pointer" }}>{d.label}</button>;
          })}
        </div>
      )}

      <div style={{ flex:1, padding:"16px 20px", overflowY:"auto" }}>
        {tab === "input" && (
          <div style={{ maxWidth:800, margin:"0 auto" }}>
            <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:12 }}>
              <span style={{ fontSize:13, color:"#aaa" }}>Documents ({docs.length})</span>
              <button onClick={addDoc} style={{ padding:"4px 12px", background:"#2a2a2a", color:"#4ecdc4", border:"1px solid #444", borderRadius:4, fontSize:12, fontFamily:"monospace", cursor:"pointer" }}>+ Add</button>
              <label style={{ padding:"4px 12px", background:"#2a2a2a", color:"#45b7d1", border:"1px solid #444", borderRadius:4, fontSize:12, fontFamily:"monospace", cursor:"pointer" }}>Upload .txt<input type="file" multiple accept=".txt" style={{ display:"none" }} onChange={function(ev) { handleFiles(ev.target.files); }} /></label>
            </div>
            <div style={{ display:"flex", gap:4, marginBottom:12, flexWrap:"wrap" }}>
              {docs.map(function(d) { return <button key={d.id} onClick={function() { setActiveInputDoc(d.id); }} style={{ padding:"5px 12px", borderRadius:4, border: "1px solid " + (activeInputDoc === d.id ? "#45b7d1" : "#333"), background: activeInputDoc === d.id ? "#1a2a2a" : "#1a1a1a", color: activeInputDoc === d.id ? "#45b7d1" : "#888", fontSize:12, fontFamily:"monospace", cursor:"pointer" }}>{d.label}{docs.length > 1 && <span onClick={function(ev) { ev.stopPropagation(); rmDoc(d.id); }} style={{ marginLeft:8, color:"#666", cursor:"pointer" }}>x</span>}</button>; })}
            </div>
            {docs.filter(function(d) { return d.id === activeInputDoc; }).map(function(d) {
              return (
                <div key={d.id}>
                  <div style={{ display:"flex", gap:8, alignItems:"center", marginBottom:8 }}>
                    <input type="text" value={d.label} onChange={function(ev) { updDoc(d.id, "label", ev.target.value); }} style={{ width:300, padding:"6px 10px", background:"#1a1a1a", border:"1px solid #444", borderRadius:4, color:"#ccc", fontSize:13, fontFamily:"monospace", boxSizing:"border-box" }} />
                    {hasMarkers && <button onClick={function() { const p = parseSep(d.text); if (p) { setDocs(p); setActiveInputDoc(p[0].id); } }} style={{ padding:"6px 14px", background:"#2a2a1a", color:"#f0b27a", border:"1px solid #f0b27a44", borderRadius:4, cursor:"pointer", fontSize:11, fontFamily:"monospace" }}>Split</button>}
                  </div>
                  <textarea value={d.text} onChange={function(ev) { updDoc(d.id, "text", ev.target.value); }} onPaste={function(ev) { const t = ev.clipboardData.getData("text"); if (t.includes("---DOC")) { ev.preventDefault(); const p = parseSep(t); if (p) { setDocs(p); setActiveInputDoc(p[0].id); } } }} placeholder="Paste text here..." style={{ width:"100%", height:"calc(100vh - 400px)", background:"#0d0d0d", border:"1px solid #2a2a2a", borderRadius:6, color:"#ccc", padding:16, fontSize:13, fontFamily:"monospace", resize:"none", boxSizing:"border-box", lineHeight:1.6 }} />
                </div>
              );
            })}
            <div style={{ marginTop:12, display:"flex", gap:8, alignItems:"center", flexWrap:"wrap" }}>
              <button onClick={function() { setShowParams(!showParams); }} style={{ padding:"6px 12px", background:"#1a1a1a", color:"#888", border:"1px solid #333", borderRadius:4, cursor:"pointer", fontSize:11, fontFamily:"monospace" }}>Parameters {showParams ? "\u25BE" : "\u25B8"}</button>
              <button onClick={runAnalysis} disabled={!validDocs.length || loading} style={{ padding:"8px 20px", background: validDocs.length && !loading ? "#4ecdc4" : "#333", color: validDocs.length && !loading ? "#111" : "#666", border:"none", borderRadius:4, cursor: validDocs.length && !loading ? "pointer" : "default", fontFamily:"monospace", fontSize:13, fontWeight:"bold" }}>{"Analyze " + validDocs.length + " doc" + (validDocs.length !== 1 ? "s" : "") + " \u2192"}</button>
              {msg && <span style={{ fontSize:10, color:"#f7dc6f" }}>{msg}</span>}
            </div>
            {showParams && (
              <div style={{ marginTop:10, padding:14, background:"#1a1a1a", borderRadius:6, border:"1px solid #333", display:"flex", gap:20, flexWrap:"wrap", alignItems:"flex-end" }}>
                <div><label style={{ fontSize:10, color:"#888", display:"block", marginBottom:3 }}>Top N: {topN}</label><input type="range" min={10} max={50} value={topN} onChange={function(ev) { setTopN(+ev.target.value); }} style={{ width:100 }} /></div>
                <div><label style={{ fontSize:10, color:"#888", display:"block", marginBottom:3 }}>WN depth: {wnDepth}</label><input type="range" min={1} max={3} value={wnDepth} onChange={function(ev) { setWnDepth(+ev.target.value); }} style={{ width:80 }} /></div>
                <div><label style={{ fontSize:10, color:"#888", display:"block", marginBottom:3 }}>Decay: {decay.toFixed(2)}</label><input type="range" min={30} max={80} value={decay * 100} onChange={function(ev) { rerunDecay(+ev.target.value / 100); }} style={{ width:100 }} /></div>
              </div>
            )}
          </div>
        )}

        {tab === "weave" && activeWR && (
          <div style={{ maxWidth:1100, margin:"0 auto" }}>
            <div style={{ display:"flex", gap:8, marginBottom:12, alignItems:"center", height:36, flexWrap:"wrap" }}>
              {LAYER_CFG.map(function(l) { return <button key={l.id} onClick={function() { toggleWLayer(l.id); }} title={l.desc} style={{ padding:"5px 10px", borderRadius:4, fontSize:11, fontFamily:"monospace", cursor:"pointer", background: wLayers[l.id] ? l.color + "22" : "#1a1a1a", color: wLayers[l.id] ? l.color : "#555", border: "1px solid " + (wLayers[l.id] ? l.color : "#333"), transition:"all 0.15s" }}>{l.label}</button>; })}
              <div style={{ width:40, flexShrink:0 }}>{wLayers.emotion && <EmoToggle enabledSlots={enabledSlots} setEnabledSlots={setEnabledSlots} />}</div>
              <div style={{ width:1, height:22, background:"#333" }} />
              <div style={{ display:"flex", border:"1px solid #333", borderRadius:4, overflow:"hidden" }}>
                {[10,20,30].map(function(g) { return <button key={g} onClick={function() { setGridSize(g); }} style={{ padding:"5px 11px", background: gridSize === g ? "#bb8fce" : "#1a1a1a", color: gridSize === g ? "#111" : "#666", border:"none", cursor:"pointer", fontSize:11, fontFamily:"monospace", fontWeight: gridSize === g ? "bold" : "normal" }}>{g + "\u00B2"}</button>; })}
              </div>
              <div style={{ marginLeft:"auto", display:"flex", gap:8, alignItems:"center" }}>
                <button onClick={function() { setShowArousal(!showArousal); }} style={{ padding:"5px 10px", background: showArousal ? "#2a2a1a" : "#1a1a1a", color: showArousal ? "#f7dc6f" : "#555", border: "1px solid " + (showArousal ? "#f7dc6f44" : "#333"), borderRadius:4, cursor:"pointer", fontSize:11, fontFamily:"monospace" }}>Arousal</button>
                <div style={{ width:1, height:22, background:"#333" }} />
                <div style={{ display:"flex", border:"1px solid #333", borderRadius:4, overflow:"hidden" }}>
                  {[["bi","Bi"],["up","\u2191"],["down","\u2193"]].map(function(p) { return <button key={p[0]} onClick={function() { rerunFlow(p[0]); }} style={{ padding:"5px 9px", background: flow === p[0] ? "#45b7d1" : "#1a1a1a", color: flow === p[0] ? "#111" : "#666", border:"none", cursor:"pointer", fontSize:11, fontFamily:"monospace" }}>{p[1]}</button>; })}
                </div>
                <div style={{ width:1, height:22, background:"#333" }} />
                <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                  <span style={{ fontSize:10, color:"#666" }}>N:</span>
                  <input type="range" min={10} max={50} value={topN} onChange={function(ev) { rerunTopN(+ev.target.value); }} style={{ width:60 }} />
                  <span style={{ fontSize:10, color:"#aaa", width:16 }}>{topN}</span>
                </div>
                <div style={{ width:1, height:22, background:"#333" }} />
                <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                  <span style={{ fontSize:10, color:"#666" }}>decay:</span>
                  <input type="range" min={30} max={80} value={decay * 100} onChange={function(ev) { rerunDecay(+ev.target.value / 100); }} style={{ width:50 }} />
                  <span style={{ fontSize:10, color:"#aaa", width:24 }}>{decay.toFixed(2)}</span>
                </div>
              </div>
            </div>

            <div ref={contentRef} style={{ display:"flex", gap:10, alignItems:"stretch", height:540, overflow:"hidden" }}>
              <WeaveReader enriched={activeWR.enriched} layers={wLayers} highlightLemma={wHighlight} maxFreq={activeWR.maxFreq} maxRel={activeWR.maxRel} onHover={function(t, x, y) { setWHovTok(t); setWHovPos({ x:x, y:y }); }} onClick={function(lem) { setWHighlight(function(prev) { return prev === lem ? null : lem; }); }} enabledSlots={enabledSlots} showArousal={showArousal} scrollRef={readerRef} onScroll={handleReaderScroll} gridSize={gridSize} />
              <WeaveMinimap enriched={activeWR.enriched} layers={wLayers} enabledSlots={enabledSlots} maxFreq={activeWR.maxFreq} maxRel={activeWR.maxRel} scrollFrac={scrollFrac} viewFrac={viewFrac} onSeek={handleMinimapSeek} height={contentH} />
              <div style={{ width:160, flexShrink:0, display:"flex", flexDirection:"column" }}>
                <div style={{ fontSize:10, color:"#888", marginBottom:3 }}>Top {topN} {"\u00B7"} click to highlight</div>
                <WeaveWordPanel result={activeWR} topN={topN} highlightLemma={wHighlight} onClickWord={function(w) { setWHighlight(w); }} ngMode={wNgMode} setNgMode={setWNgMode} sortBy={wSortBy} setSortBy={setWSortBy} />
              </div>
            </div>

            <div style={{ display:"flex", gap:14, marginTop:10, padding:"8px 12px", background:"#0d0d0d", borderRadius:4, border:"1px solid #1a1a1a", fontSize:11, color:"#555", flexWrap:"wrap" }}>
              {wLayers.polarity && <span><span style={{ color:"#82e0aa" }}>{"\u25A0"}</span>/<span style={{ color:"#ff6b6b" }}>{"\u25A0"}</span> polarity</span>}
              {wLayers.emotion && <span><span style={{ color:"#f0b27a" }}>{"\u2014"}</span> emotion</span>}
              {showArousal && <span><span style={{ color:"#f7dc6f" }}>{"\u2501"}</span> arousal</span>}
              {wLayers.frequency && <span><span style={{ color:"#bb8fce" }}>{"\u25CB"}</span> brightness</span>}
              {wLayers.relevance && <span><span style={{ color:"#4ecdc4", fontWeight:600 }}>B</span> weight</span>}
              {wLayers.community && <span style={{ background:"#4ecdc41a", padding:"0 4px", borderRadius:2 }}>community</span>}
            </div>
          </div>
        )}

        {tab === "weave" && !activeWR && (
          <div style={{ flex:1, display:"flex", alignItems:"center", justifyContent:"center" }}>
            <div style={{ textAlign:"center", color:"#555" }}>
              <div style={{ fontSize:14, marginBottom:8 }}>{"\u2190"} Analyze documents first.</div>
              <div style={{ fontSize:11, color:"#444" }}>Six analytical layers projected onto the text itself.</div>
            </div>
          </div>
        )}
      </div>

      <WeaveTooltip token={wHovTok} x={wHovPos.x} y={wHovPos.y} />
      {loading && <div style={{ position:"fixed", bottom:20, left:"50%", transform:"translateX(-50%)", padding:"8px 20px", background:"#1a1a1aee", border:"1px solid #444", borderRadius:6, fontSize:11, color:"#f7dc6f", fontFamily:"monospace", zIndex:999 }}>{msg || "Processing..."}</div>}
    </div>
  );
}
