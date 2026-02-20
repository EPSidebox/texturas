const { useState, useRef, useEffect, useCallback, useMemo } = React;

const ASSET_BASE_URL = "https://epsidebox.github.io/texturas/assets/";
const STOP_WORDS = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","shall","should","may","might","can","could","this","that","these","those","it","its","i","me","my","mine","we","us","our","ours","you","your","yours","he","him","his","she","her","hers","they","them","their","theirs","what","which","who","whom","whose","where","when","how","why","if","then","else","so","because","as","than","too","very","just","about","above","after","again","all","also","am","any","before","below","between","both","during","each","few","further","get","got","here","into","more","most","much","only","other","out","over","own","same","some","such","there","through","under","until","up","while","down","like","make","many","now","off","once","one","onto","per","since","still","take","thing","things","think","two","upon","well","went","yeah","yes","oh","um","uh","ah","okay","ok","really","actually","basically","literally","right","know","going","go","come","say","said","tell","told","see","look","want","need","way","back","even","around","something","anything","everything","someone","anyone","everyone","always","sometimes","often","already","yet","kind","sort","lot","bit","little","big","good","great","long","new","old","first","last","next","time","times","day","days","year","years","people","person","part","place","point","work","life","world","hand","week","end","case","fact","area","number","given","able","made","set","used","using","another","different","example","enough","however","important","include","including","keep","large","whether","without","within","along","become","across","among","toward","towards","though","although","either","rather","several","certain","less","likely","began","begun","brought","thus","therefore","hence","moreover","furthermore","nevertheless","nonetheless","meanwhile","accordingly","consequently","subsequently"]);
const NEGATION = new Set(["not","no","never","neither","nor","don't","doesn't","didn't","won't","wouldn't","can't","couldn't","shouldn't","isn't","aren't","wasn't","weren't","haven't","hasn't","hadn't","cannot","nothing","nobody","none","nowhere"]);
const EMOTIONS = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"];
const EC = {anger:"#ff6b6b",anticipation:"#f7dc6f",disgust:"#bb8fce",fear:"#85c1e9",joy:"#82e0aa",sadness:"#45b7d1",surprise:"#f0b27a",trust:"#4ecdc4"};
const EMO_LAYOUT = [
  {r:0,c:0,emo:"anticipation"},{r:0,c:1,emo:"joy"},{r:0,c:2,emo:"trust"},
  {r:1,c:0,emo:"anger"},{r:1,c:1,emo:null},{r:1,c:2,emo:"fear"},
  {r:2,c:0,emo:"disgust"},{r:2,c:1,emo:"sadness"},{r:2,c:2,emo:"surprise"}
];

const WORLD=10,HALF=5,FRU=6.5;
const ISO_ELEV=Math.atan(1/Math.sqrt(2)),ISO_AZI=-Math.PI/4,CAM_D=80,DIM_OP=0.06;
const VOL_MUL={10:5,20:3.5,30:2};
const EMO_VOL_CAP=0.5;
const EMO_BRIGHT_K=15;

function cs(gs){return WORLD/gs}
function camFlat(){return{pos:new THREE.Vector3(0,CAM_D,0.001),up:new THREE.Vector3(0,0,-1)}}
function camIso(){return{pos:new THREE.Vector3(CAM_D*Math.cos(ISO_ELEV)*Math.sin(ISO_AZI),CAM_D*Math.sin(ISO_ELEV),CAM_D*Math.cos(ISO_ELEV)*Math.cos(ISO_AZI)),up:new THREE.Vector3(0,1,0)}}
function emoBright(p){return p>0?(Math.log(1+p*EMO_BRIGHT_K)/Math.log(1+EMO_BRIGHT_K)):0}

class POSTagger{constructor(){this.lk=null;this.ready=false}load(d){this.lk=d;this.ready=true}
sfx=[["tion","n"],["sion","n"],["ment","n"],["ness","n"],["ity","n"],["ance","n"],["ence","n"],["ism","n"],["ist","n"],["ting","v"],["sing","v"],["ning","v"],["ing","v"],["ated","v"],["ized","v"],["ed","v"],["ize","v"],["ify","v"],["ously","r"],["ively","r"],["fully","r"],["ally","r"],["ly","r"],["ful","a"],["ous","a"],["ive","a"],["able","a"],["ible","a"],["ical","a"],["less","a"],["al","a"]];
tag(w){const l=w.toLowerCase();if(this.lk?.[l])return this.lk[l];for(const[s,p]of this.sfx)if(l.endsWith(s))return p;return"n"}}
class Lemmatizer{constructor(){this.exc=null;this.rl=null;this.ready=false}load(d){this.exc=d.exceptions;this.rl=d.rules;this.ready=true}
lemmatize(w,pos){const l=w.toLowerCase();if(this.exc?.[pos]?.[l])return this.exc[pos][l];for(const[s,r]of(this.rl?.[pos]||[]))if(l.endsWith(s)&&l.length>s.length)return l.slice(0,-s.length)+r;return l}}
class SynsetEngine{constructor(){this.d=null;this.ready=false}load(d){this.d=d;this.ready=true}
getRels(w,p){const e=this.d?.[`${w}#${p}`];if(!e)return{h:[],y:[],m:[]};return{h:e.hypernyms||[],y:e.hyponyms||[],m:e.meronyms||[]}}}
class SentimentScorer{constructor(){this.el=null;this.int=null;this.vad=null;this.vdr=null;this.swn=null;this.ready=false}
lEl(d){this.el=d;this._c()}lInt(d){this.int=d;this._c()}lVad(d){this.vad=d;this._c()}lVdr(d){this.vdr=d;this._c()}lSwn(d){this.swn=d;this._c()}
_c(){this.ready=!!(this.el||this.int||this.vad||this.vdr||this.swn)}
score(lem,pos){const r={lemma:lem,pos};if(this.el?.[lem])r.emolex=this.el[lem];if(this.int?.[lem])r.intensity=this.int[lem];if(this.vad?.[lem])r.vad=this.vad[lem];
if(this.vdr?.[lem]!==undefined)r.vader=this.vdr[lem];const k=`${lem}#${pos}`;if(this.swn?.[k])r.swn=this.swn[k];
else{const vs=["n","v","a","r"].map(p=>this.swn?.[`${lem}#${p}`]).filter(Boolean);if(vs.length)r.swn=vs[0]}
r.has=!!(r.emolex||r.intensity||r.vad||r.vader!==undefined||r.swn);return r}}
class VectorEngine{constructor(){this.v=new Map();this.dim=0;this.vocab=0}
async loadBin(vj,vb){const{dim,vocab}=vj;this.dim=dim;const f=new Float32Array(vb);for(const[w,i]of Object.entries(vocab))this.v.set(w,f.slice(i*dim,(i+1)*dim));this.vocab=this.v.size;return{vocab:this.vocab,dim:this.dim}}
has(w){return this.v.has(w)}isLoaded(){return this.vocab>0}}

function spreadAct(fMap,syn,pos,depth,decay,flow){
  if(!syn.ready)return{...fMap};const corpus=new Set(Object.keys(fMap)),scores={};
  for(const l of corpus)scores[l]=fMap[l]||0;
  for(const[l,f]of Object.entries(fMap)){if(f===0)continue;const p=pos.ready?pos.tag(l):"n";
    let front=[{w:l,p}],vis=new Set([l]);
    for(let h=1;h<=depth;h++){const nf=[];for(const nd of front){const r=syn.getRels(nd.w,nd.p);
      const tgts=flow==="up"?r.h:flow==="down"?r.y:[...r.h,...r.y,...r.m];
      for(const t of tgts){if(vis.has(t))continue;vis.add(t);const amt=f*Math.pow(decay,h);
        if(corpus.has(t))scores[t]=(scores[t]||0)+amt;nf.push({w:t,p:pos.ready?pos.tag(t):"n"})}}front=nf}}
  return scores}

const tokenize=t=>t.toLowerCase().replace(/[^\w\s'-]/g," ").split(/\s+/).filter(t=>t.length>1);
const getFreqs=ts=>_.chain(ts).countBy().toPairs().sortBy(1).reverse().value();
const getNg=(ts,n)=>{const g=[];for(let i=0;i<=ts.length-n;i++)g.push(ts.slice(i,i+n).join(" "));return _.chain(g).countBy().toPairs().sortBy(1).reverse().value()};

function analyzeForVellum(text,eng,topN,wnDepth,decay,flow){
  const toks=tokenize(text);
  const allLem=toks.map(t=>{const p=eng.pos.ready?eng.pos.tag(t):"n";const lemma=eng.lem.ready?eng.lem.lemmatize(t,p):t;return{surface:t,lemma,pos:p,stop:STOP_WORDS.has(t)&&!NEGATION.has(t)}});
  const filtLem=allLem.filter(t=>!t.stop).map(t=>t.lemma);
  const freqs=getFreqs(filtLem),fMap=Object.fromEntries(freqs),topWords=freqs.slice(0,topN);
  const relevance=spreadAct(fMap,eng.syn,eng.pos,wnDepth,decay,flow);
  const ng2=getNg(filtLem,2).slice(0,topN),ng3=getNg(filtLem,3).slice(0,topN);
  const ng2Map=Object.fromEntries(ng2),ng3Map=Object.fromEntries(ng3);
  const enriched=allLem.map(t=>{const s=eng.sent.ready?eng.sent.score(t.lemma,t.pos):{};
    return{...t,rel:relevance[t.lemma]||0,freq:fMap[t.lemma]||0,vader:s.vader!==undefined?s.vader:null,
      arousal:s.vad?s.vad.a:null,emolex:s.emolex||null}});
  return{enriched,freqs,fMap,topWords,relevance,tw:toks.length,ng2,ng3,ng2Map,ng3Map}}

function vellumBins(enriched,gs,filterSet,ngMode,ngFreqMap){
  const cells=gs*gs,tw=enriched.length,hasFilter=filterSet.size>0,bins=[];
  const ngN=ngMode==="bigrams"?2:ngMode==="trigrams"?3:1;const isNg=ngN>1;
  const filtPos=[];for(let i=0;i<enriched.length;i++){if(!enriched[i].stop)filtPos.push({idx:i,lemma:enriched[i].lemma})}
  let ngOcc=[];
  if(isNg){for(let i=0;i<=filtPos.length-ngN;i++){const gram=[];for(let k=0;k<ngN;k++)gram.push(filtPos[i+k].lemma);
    const key=gram.join(" ");if(ngFreqMap?.[key])ngOcc.push({startIdx:filtPos[i].idx,key,freq:ngFreqMap[key]})}}
  let ngramPos=null;
  if(hasFilter&&isNg){ngramPos=new Set();for(const occ of ngOcc){if(!filterSet.has(occ.key))continue;
    const fi=filtPos.findIndex(fp=>fp.idx===occ.startIdx);if(fi<0)continue;for(let k=0;k<ngN&&fi+k<filtPos.length;k++)ngramPos.add(filtPos[fi+k].idx)}}
  function ngFR(s,e){const fs=[];for(const o of ngOcc){if(o.startIdx>=s&&o.startIdx<e)fs.push(o.freq)}return fs}
  function computeEmo(match){const ed={};EMOTIONS.forEach(e=>{const wE=match.filter(t=>t.emolex&&t.emolex[e]);
    const aros=wE.map(t=>t.arousal).filter(v=>v!==null);
    ed[e]={prop:match.length?wE.length/match.length:0,aro:aros.length?_.mean(aros):0}});return ed}
  if(tw<=cells){for(let i=0;i<cells;i++){const tok=enriched[i];
    if(!tok){bins.push({i,empty:true,w:0,words:new Set(),rel:0,vader:0,arousal:0,topW:[],dimmed:false,wList:[],fullText:[],emo:{}});continue}
    const words=new Set(tok.stop?[]:[tok.lemma]);let dimmed=false,match=[];
    if(hasFilter){if(!isNg){match=filterSet.has(tok.lemma)?[tok]:[];dimmed=match.length===0}
      else{match=ngramPos&&ngramPos.has(i)?[tok]:[];dimmed=match.length===0}}
    else{match=tok.stop?[]:[tok]}
    let relVal;if(isNg){const nf=ngFR(i,i+1);relVal=nf.length?_.mean(nf):0}else{relVal=match.length?match[0].rel:0}
    bins.push({i,empty:false,dimmed,w:1,words,rel:relVal,
      vader:match.length&&match[0].vader!==null?match[0].vader:0,
      arousal:match.length&&match[0].arousal!==null?match[0].arousal:0,
      topW:match.filter(t=>t.freq>0).map(t=>({lemma:t.lemma,freq:t.freq})),
      wList:tok.stop?[]:[tok.lemma],fullText:[{surface:tok.surface,lemma:tok.lemma,stop:tok.stop}],
      emo:computeEmo(match)})}
  }else{const base=Math.floor(tw/cells),extra=tw%cells;let pos=0;
    for(let i=0;i<cells;i++){const sz=i<extra?base+1:base;const chunk=enriched.slice(pos,pos+sz);const sp=pos;pos+=sz;
      const words=new Set(chunk.filter(t=>!t.stop).map(t=>t.lemma));let match,dimmed=false;
      if(hasFilter){if(!isNg){match=chunk.filter(t=>filterSet.has(t.lemma));dimmed=match.length===0}
        else{match=[];for(let j=0;j<chunk.length;j++){if(ngramPos&&ngramPos.has(sp+j)&&!chunk[j].stop)match.push(chunk[j])}dimmed=match.length===0}}
      else{match=chunk.filter(t=>!t.stop)}
      const vdrs=match.map(t=>t.vader).filter(v=>v!==null),aros=match.map(t=>t.arousal).filter(v=>v!==null);
      const topW=_.chain(match).filter(t=>t.freq>0).sortBy(t=>-t.freq).uniqBy("lemma").take(5).value();
      let relVal;if(isNg){const nf=ngFR(sp,sp+sz);relVal=nf.length?_.mean(nf):0}else{const rels=match.map(t=>t.rel).filter(v=>v>0);relVal=rels.length?_.mean(rels):0}
      bins.push({i,empty:false,dimmed,w:chunk.length,words,rel:relVal,
        vader:vdrs.length?_.mean(vdrs):0,arousal:aros.length?_.mean(aros):0,topW,
        wList:chunk.filter(t=>!t.stop).map(t=>t.lemma),
        fullText:chunk.map(t=>({surface:t.surface,lemma:t.lemma,stop:t.stop})),
        emo:computeEmo(match)})}}
  return{bins,bs:tw<=cells?1:Math.ceil(tw/cells),cells}}

function normArr(v,m,mx){const x=mx!==undefined?mx:Math.max(...v.map(Math.abs));if(x===0)return v.map(()=>0);if(m==="log"){const l=Math.log(1+x);return v.map(a=>Math.sign(a)*Math.log(1+Math.abs(a))/l)}return v.map(a=>a/x)}

const DB="texturas-cache",DBV=1,STO="assets";
function openDB(){return new Promise((r,j)=>{const q=indexedDB.open(DB,DBV);q.onupgradeneeded=()=>q.result.createObjectStore(STO);q.onsuccess=()=>r(q.result);q.onerror=()=>j(q.error)})}
async function cGet(k){try{const db=await openDB();return new Promise(r=>{const t=db.transaction(STO,"readonly");const q=t.objectStore(STO).get(k);q.onsuccess=()=>r(q.result||null);q.onerror=()=>r(null)})}catch{return null}}
async function cSet(k,v){try{const db=await openDB();return new Promise(r=>{const t=db.transaction(STO,"readwrite");t.objectStore(STO).put(v,k);t.oncomplete=()=>r(true);t.onerror=()=>r(false)})}catch{return false}}
async function loadAsset(key,path,bin,cb){const c=await cGet(key);if(c)return c;if(!ASSET_BASE_URL)return null;const b=ASSET_BASE_URL.endsWith("/")?ASSET_BASE_URL:ASSET_BASE_URL+"/";if(cb)cb(`Fetching ${path}...`);try{const r=await fetch(b+path);if(!r.ok)return null;const d=bin?await r.arrayBuffer():await r.json();await cSet(key,d);return d}catch{return null}}

function buildScene(W,H,gs,isVol){
  const sc=new THREE.Scene();sc.background=new THREE.Color(0x0d0d0d);
  const asp=W/H,cam=new THREE.OrthographicCamera(-FRU*asp,FRU*asp,FRU,-FRU,0.1,200);
  const init=isVol?camIso():camFlat();cam.position.copy(init.pos);cam.up.copy(init.up);cam.lookAt(0,0,0);
  const ren=new THREE.WebGLRenderer({antialias:true});ren.setSize(W,H);ren.setPixelRatio(Math.min(window.devicePixelRatio,2));
  sc.add(new THREE.AmbientLight(0xffffff,0.45));
  const dl=new THREE.DirectionalLight(0xffffff,0.85);dl.position.set(-5,15,-8);sc.add(dl);
  const dl2=new THREE.DirectionalLight(0x8899bb,0.3);dl2.position.set(8,10,6);sc.add(dl2);
  const c=cs(gs),gm=new THREE.LineBasicMaterial({color:0x1a1a1a});
  for(let i=0;i<=gs;i++){const p=i*c-HALF;
    sc.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(p,-.01,-HALF),new THREE.Vector3(p,-.01,HALF)]),gm));
    sc.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-HALF,-.01,p),new THREE.Vector3(HALF,-.01,p)]),gm))}
  const pl=new THREE.Mesh(new THREE.PlaneGeometry(WORLD+.2,WORLD+.2),new THREE.MeshPhongMaterial({color:0x0f0f0f}));pl.rotation.x=-Math.PI/2;pl.position.y=-.02;sc.add(pl);
  return{sc,cam,ren,c}}

function cleanup(el,S,fr){
  cancelAnimationFrame(fr.current);
  if(S._onR)window.removeEventListener("resize",S._onR);
  if(S.ren?.domElement){if(S._onMv)S.ren.domElement.removeEventListener("mousemove",S._onMv);if(S._onClick)S.ren.domElement.removeEventListener("click",S._onClick)}
  if(S.ren&&el&&el.contains(S.ren.domElement))el.removeChild(S.ren.domElement);
  if(S.ren)S.ren.dispose()}

// ── Shared tooltip ──
function VellumTooltip({ai,bins,gs,pinned,topNSet,showSize,showColor,showVol,ngMode,onDismiss}){
  const ab=ai!==null?bins[ai]:null;if(!ab||ab.empty||ab.dimmed)return null;
  const ip=pinned;const nm=ngMode||"words";
  const emos=EMOTIONS.filter(e=>ab.emo[e]?.prop>0);
  return(<div onClick={e=>{if(ip){e.stopPropagation();onDismiss()}}} style={{position:"absolute",top:8,left:8,padding:"8px 12px",background:ip?"#111111ee":"#111111aa",border:`1px solid ${ip?"#4ecdc4":"#333"}`,borderRadius:6,fontFamily:"monospace",fontSize:10,pointerEvents:ip?"auto":"none",maxWidth:ip?300:220,maxHeight:ip?300:120,overflowY:ip?"auto":"hidden",zIndex:3,cursor:ip?"pointer":"default",backdropFilter:"blur(2px)"}}>
    <div style={{color:"#4ecdc4",fontSize:12,marginBottom:3}}>[{Math.floor(ai/gs)+1},{(ai%gs)+1}] <span style={{color:"#555",fontSize:9}}>{((ai/(gs*gs))*100).toFixed(1)}% · {ab.w}w</span>{ip&&<span style={{color:"#555",fontSize:9}}> · dismiss</span>}</div>
    <div style={{color:"#888"}}>Rel: <span style={{color:"#4ecdc4"}}>{ab.rel.toFixed(2)}</span> · V: <span style={{color:ab.vader>.05?"#82e0aa":ab.vader<-.05?"#ff6b6b":"#666"}}>{ab.vader>0?"+":""}{ab.vader.toFixed(3)}</span> · A: <span style={{color:ab.arousal>0?"#f7dc6f":ab.arousal<0?"#45b7d1":"#666"}}>{ab.arousal>0?"+":""}{ab.arousal.toFixed(3)}</span></div>
    {emos.length>0&&<div style={{display:"flex",flexWrap:"wrap",gap:3,marginTop:3}}>{emos.map(e=>(<span key={e} style={{fontSize:9,color:EC[e]}}>{e.slice(0,3)}:{(ab.emo[e].prop*100).toFixed(0)}%</span>))}</div>}
    {ip&&ab.fullText&&(<div style={{marginTop:5,borderTop:"1px solid #2a2a2a",paddingTop:4,fontSize:10,lineHeight:1.6}}>{Array.isArray(ab.fullText)&&typeof ab.fullText[0]==="object"?ab.fullText.map((t,j)=>{const h=topNSet.has(t.lemma);return(<span key={j}>{j>0?" ":""}<span style={{color:t.stop?"#444":h?"#4ecdc4":"#888",fontWeight:h?"bold":"normal"}}>{t.surface}</span></span>)}):<span style={{color:"#888"}}>{String(ab.fullText)}</span>}</div>)}
  </div>)}

// ── CHANNELS ──
function VellumGrid({bins,scale,showSize,showColor,showVol,gridSize:gs,normMaxes,label,ngMode,fixedWidth,fixedHeight,topNWords,pinnedIdx,onPin}){
  const cRef=useRef(),stRef=useRef({}),frameRef=useRef();
  const[hov,setHov]=useState(null);
  const svRef=useRef(showVol);svRef.current=showVol;
  const topNSet=useMemo(()=>new Set((topNWords||[]).map(([w])=>w)),[topNWords]);
  const sizeV=useMemo(()=>normArr(bins.map(b=>b.rel),scale,normMaxes?.rel),[bins,scale,normMaxes]);
  const colV=useMemo(()=>bins.map(b=>b.vader),[bins]);
  const volV=useMemo(()=>normArr(bins.map(b=>b.arousal),scale,normMaxes?.arousal),[bins,scale,normMaxes]);

  useEffect(()=>{
    const el=cRef.current;if(!el)return;const S=stRef.current;
    if(S.ren)cleanup(el,S,frameRef);
    let dead=false;
    const build=(W,H)=>{if(dead||W<10||H<10)return;
      const{sc,cam,ren,c}=buildScene(W,H,gs,svRef.current);
      el.appendChild(ren.domElement);
      const geo=new THREE.BoxGeometry(c*.9,1,c*.9),boxes=[];
      for(let i=0;i<gs*gs;i++){const mat=new THREE.MeshPhongMaterial({color:0x4ecdc4,transparent:true,opacity:.85,shininess:50});const mesh=new THREE.Mesh(geo,mat);
        mesh.position.set((i%gs)*c-HALF+c/2,0,Math.floor(i/gs)*c-HALF+c/2);
        mesh.userData={idx:i};sc.add(mesh);boxes.push(mesh)}
      const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();
      S.sc=sc;S.cam=cam;S.ren=ren;S.boxes=boxes;S.ray=ray;S.mouse=mouse;S.tPos=cam.position.clone();S.tUp=cam.up.clone();
      const onMv=e=>{const r=ren.domElement.getBoundingClientRect();mouse.x=((e.clientX-r.left)/r.width)*2-1;mouse.y=-((e.clientY-r.top)/r.height)*2+1;ray.setFromCamera(mouse,cam);const h=ray.intersectObjects(boxes);setHov(h.length?h[0].object.userData.idx:null);ren.domElement.style.cursor=h.length?"crosshair":"default"};
      const onClick=e=>{const r=ren.domElement.getBoundingClientRect();mouse.x=((e.clientX-r.left)/r.width)*2-1;mouse.y=-((e.clientY-r.top)/r.height)*2+1;ray.setFromCamera(mouse,cam);const h=ray.intersectObjects(boxes);if(h.length){const idx=h[0].object.userData.idx;onPin(pinnedIdx===idx?null:idx)}else onPin(null)};
      ren.domElement.addEventListener("mousemove",onMv);ren.domElement.addEventListener("click",onClick);S._onMv=onMv;S._onClick=onClick;
      const tick=()=>{if(dead)return;frameRef.current=requestAnimationFrame(tick);cam.position.lerp(S.tPos,.06);cam.up.lerp(S.tUp,.06);cam.lookAt(0,0,0);cam.updateProjectionMatrix();ren.render(sc,cam)};tick();
      const onR=()=>{const w=el.clientWidth,h=el.clientHeight;if(w<10||h<10)return;const a=w/h;cam.left=-FRU*a;cam.right=FRU*a;cam.top=FRU;cam.bottom=-FRU;cam.updateProjectionMatrix();ren.setSize(w,h)};
      window.addEventListener("resize",onR);S._onR=onR};
    if(fixedWidth&&fixedHeight)requestAnimationFrame(()=>build(fixedWidth,fixedHeight));
    else{const ro=new ResizeObserver(en=>{const{width:w,height:h}=en[0].contentRect;if(w>10&&h>10){ro.disconnect();build(Math.round(w),Math.round(h))}});ro.observe(el);return()=>{dead=true;ro.disconnect();cleanup(el,S,frameRef)}}
    return()=>{dead=true;cleanup(el,S,frameRef)};
  },[gs,fixedWidth,fixedHeight]);

  // Click handler reads pinnedIdx via ref to avoid stale closure
  const piRef=useRef(pinnedIdx);piRef.current=pinnedIdx;
  useEffect(()=>{const S=stRef.current;if(!S.ren?.domElement)return;
    const onClick=e=>{const r=S.ren.domElement.getBoundingClientRect();S.mouse.x=((e.clientX-r.left)/r.width)*2-1;S.mouse.y=-((e.clientY-r.top)/r.height)*2+1;S.ray.setFromCamera(S.mouse,S.cam);const h=S.ray.intersectObjects(S.boxes);if(h.length){const idx=h[0].object.userData.idx;onPin(piRef.current===idx?null:idx)}else onPin(null)};
    if(S._onClick)S.ren.domElement.removeEventListener("click",S._onClick);
    S.ren.domElement.addEventListener("click",onClick);S._onClick=onClick;
  },[onPin]);

  useEffect(()=>{const S=stRef.current;if(!S.boxes||S.boxes.length!==bins.length)return;
    const vm=VOL_MUL[gs]||5;
    S.boxes.forEach((mesh,i)=>{const b=bins[i];if(b.empty){mesh.visible=false;return}mesh.visible=true;
      const sz=showSize?Math.max(.04,sizeV[i]):1;
      if(showVol){const raw=volV[i];const h=Math.max(.08,Math.abs(raw)*vm);mesh.scale.set(sz,b.dimmed?.04:h,sz);mesh.position.y=b.dimmed?.02:raw>=0?h/2:-h/2}
      else{mesh.scale.set(sz,.06,sz);mesh.position.y=.03}
      if(b.dimmed){mesh.material.opacity=DIM_OP;mesh.material.color.set(0x222222)}
      else if(showColor){const v=colV[i]/(normMaxes?.vader||.01);if(v>.05)mesh.material.color.setHSL(.38,.65,.3+Math.min(v,1)*.35);else if(v<-.05)mesh.material.color.setHSL(0,.65,.3+Math.min(Math.abs(v),1)*.35);else mesh.material.color.set(0x444444);mesh.material.opacity=showSize?.15+sizeV[i]*.8:.85}
      else{mesh.material.color.set(0x4ecdc4);mesh.material.opacity=showSize?.15+sizeV[i]*.8:.85}
      if((hov===i||pinnedIdx===i)&&!b.dimmed){mesh.material.emissive.set(0xffffff);mesh.material.emissiveIntensity=pinnedIdx===i?.45:.35}else{mesh.material.emissive.set(0);mesh.material.emissiveIntensity=0}})},
  [sizeV,colV,volV,showSize,showColor,showVol,hov,pinnedIdx,bins,gs,normMaxes]);

  useEffect(()=>{const S=stRef.current;if(!S.cam)return;const t=showVol?camIso():camFlat();S.tPos.copy(t.pos);S.tUp.copy(t.up)},[showVol]);

  const ai=pinnedIdx!==null?pinnedIdx:hov;
  return(<div style={{position:"relative",width:"100%",height:"100%"}}>{label&&<div style={{position:"absolute",top:8,right:10,fontSize:11,color:"#45b7d1",fontFamily:"monospace",zIndex:2,pointerEvents:"none"}}>{label}</div>}
    <div ref={cRef} style={{width:"100%",height:"100%",minHeight:200,borderRadius:6,overflow:"hidden",border:"1px solid #2a2a2a"}}/>
    <VellumTooltip ai={ai} bins={bins} gs={gs} pinned={pinnedIdx!==null} topNSet={topNSet} showSize={showSize} showColor={showColor} showVol={showVol} ngMode={ngMode} onDismiss={()=>onPin(null)}/>
    <div style={{position:"absolute",bottom:6,left:8,fontSize:9,color:"#333",fontFamily:"monospace"}}>{gs}²</div>
  </div>)}

// ── EMOTIONS ──
function VellumEmoGrid({bins,scale,showVol,gridSize:gs,normMaxes,label,fixedWidth,fixedHeight,enabledSlots,topNWords,pinnedIdx,onPin}){
  const cRef=useRef(),stRef=useRef({}),frameRef=useRef();
  const[hov,setHov]=useState(null);
  const svRef=useRef(showVol);svRef.current=showVol;
  const topNSet=useMemo(()=>new Set((topNWords||[]).map(([w])=>w)),[topNWords]);
  const relV=useMemo(()=>normArr(bins.map(b=>b.rel),scale,normMaxes?.rel),[bins,scale,normMaxes]);

  useEffect(()=>{
    const el=cRef.current;if(!el)return;const S=stRef.current;
    if(S.ren)cleanup(el,S,frameRef);
    let dead=false;
    const build=(W,H)=>{if(dead||W<10||H<10)return;
      const{sc,cam,ren,c}=buildScene(W,H,gs,svRef.current);
      el.appendChild(ren.domElement);
      const subSz=c*0.28,subSp=c*0.33,geo=new THREE.BoxGeometry(subSz,1,subSz);
      const cellMeshes=[];
      for(let ci=0;ci<gs*gs;ci++){const cr=Math.floor(ci/gs),cc=ci%gs;
        const cx=cc*c-HALF+c/2,cz=cr*c-HALF+c/2;
        const subs=[];
        EMO_LAYOUT.forEach(({r,c:col,emo},si)=>{
          const ox=(col-1)*subSp,oz=(r-1)*subSp;
          const baseColor=emo?new THREE.Color(EC[emo]):new THREE.Color(0xffffff);
          const mat=new THREE.MeshPhongMaterial({color:0x222222,transparent:true,opacity:.9,shininess:40});
          const mesh=new THREE.Mesh(geo,mat);
          mesh.position.set(cx+ox,0,cz+oz);mesh.scale.set(1,.06,1);
          mesh.userData={cellIdx:ci,slotIdx:si,emo,baseColor};sc.add(mesh);subs.push({slot:si,emo,mesh})});
        cellMeshes.push(subs)}
      const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();
      const allMeshes=cellMeshes.flat().map(s=>s.mesh);
      S.sc=sc;S.cam=cam;S.ren=ren;S.cellMeshes=cellMeshes;S.ray=ray;S.mouse=mouse;S.tPos=cam.position.clone();S.tUp=cam.up.clone();S.allMeshes=allMeshes;
      const onMv=e=>{const r=ren.domElement.getBoundingClientRect();mouse.x=((e.clientX-r.left)/r.width)*2-1;mouse.y=-((e.clientY-r.top)/r.height)*2+1;ray.setFromCamera(mouse,cam);const h=ray.intersectObjects(allMeshes);setHov(h.length?h[0].object.userData.cellIdx:null);ren.domElement.style.cursor=h.length?"crosshair":"default"};
      ren.domElement.addEventListener("mousemove",onMv);S._onMv=onMv;
      const tick=()=>{if(dead)return;frameRef.current=requestAnimationFrame(tick);cam.position.lerp(S.tPos,.06);cam.up.lerp(S.tUp,.06);cam.lookAt(0,0,0);cam.updateProjectionMatrix();ren.render(sc,cam)};tick();
      const onR=()=>{const w=el.clientWidth,h=el.clientHeight;if(w<10||h<10)return;const a=w/h;cam.left=-FRU*a;cam.right=FRU*a;cam.top=FRU;cam.bottom=-FRU;cam.updateProjectionMatrix();ren.setSize(w,h)};
      window.addEventListener("resize",onR);S._onR=onR};
    if(fixedWidth&&fixedHeight)requestAnimationFrame(()=>build(fixedWidth,fixedHeight));
    else{const ro=new ResizeObserver(en=>{const{width:w,height:h}=en[0].contentRect;if(w>10&&h>10){ro.disconnect();build(Math.round(w),Math.round(h))}});ro.observe(el);return()=>{dead=true;ro.disconnect();cleanup(el,S,frameRef)}}
    return()=>{dead=true;cleanup(el,S,frameRef)};
  },[gs,fixedWidth,fixedHeight]);

  // Click handler via ref for fresh pinnedIdx
  const piRef=useRef(pinnedIdx);piRef.current=pinnedIdx;
  useEffect(()=>{const S=stRef.current;if(!S.ren?.domElement)return;
    const onClick=e=>{const r=S.ren.domElement.getBoundingClientRect();S.mouse.x=((e.clientX-r.left)/r.width)*2-1;S.mouse.y=-((e.clientY-r.top)/r.height)*2+1;S.ray.setFromCamera(S.mouse,S.cam);const h=S.ray.intersectObjects(S.allMeshes);if(h.length){const idx=h[0].object.userData.cellIdx;onPin(piRef.current===idx?null:idx)}else onPin(null)};
    if(S._onClick)S.ren.domElement.removeEventListener("click",S._onClick);
    S.ren.domElement.addEventListener("click",onClick);S._onClick=onClick;
  },[onPin]);

  useEffect(()=>{const S=stRef.current;if(!S.cellMeshes)return;
    const aroMax=normMaxes?.arousal||1;
    S.cellMeshes.forEach((subs,ci)=>{const b=bins[ci];
      subs.forEach(({slot,emo,mesh})=>{
        if(b.empty){mesh.visible=false;return}
        const isCenter=emo===null;const slotKey=isCenter?"center":emo;
        const enabled=enabledSlots.has(slotKey);
        if(b.dimmed){mesh.visible=true;mesh.material.color.set(0x222222);mesh.material.opacity=DIM_OP;mesh.scale.y=.04;mesh.position.y=.02;mesh.material.emissive.set(0);mesh.material.emissiveIntensity=0;return}
        if(!enabled){mesh.visible=true;mesh.material.color.set(0x555555);mesh.material.opacity=.2;mesh.scale.y=.04;mesh.position.y=.02;mesh.material.emissive.set(0);mesh.material.emissiveIntensity=0;return}
        mesh.visible=true;
        let brightness,aro;
        if(isCenter){brightness=relV[ci];aro=b.arousal/aroMax}
        else{const ed=b.emo[emo];brightness=ed?ed.prop:0;aro=ed?(ed.aro/aroMax):0}
        const vb=isCenter?brightness:emoBright(brightness);
        const bc=mesh.userData.baseColor;
        mesh.material.color.setRGB(bc.r*vb*.85+.05,bc.g*vb*.85+.05,bc.b*vb*.85+.05);
        mesh.material.opacity=.3+vb*.65;
        if(showVol){const h=Math.min(EMO_VOL_CAP,Math.max(.06,Math.abs(aro)*3));mesh.scale.y=h;mesh.position.y=aro>=0?h/2:-h/2}
        else{mesh.scale.y=.06;mesh.position.y=.03}
        if((hov===ci||pinnedIdx===ci)&&!b.dimmed){mesh.material.emissive.set(0xffffff);mesh.material.emissiveIntensity=pinnedIdx===ci?.3:.2}
        else{mesh.material.emissive.set(0);mesh.material.emissiveIntensity=0}})})},
  [bins,relV,showVol,hov,pinnedIdx,enabledSlots,gs,normMaxes]);

  useEffect(()=>{const S=stRef.current;if(!S.cam)return;const t=showVol?camIso():camFlat();S.tPos.copy(t.pos);S.tUp.copy(t.up)},[showVol]);

  const ai=pinnedIdx!==null?pinnedIdx:hov;
  return(<div style={{position:"relative",width:"100%",height:"100%"}}>{label&&<div style={{position:"absolute",top:8,right:10,fontSize:11,color:"#45b7d1",fontFamily:"monospace",zIndex:2,pointerEvents:"none"}}>{label}</div>}
    <div ref={cRef} style={{width:"100%",height:"100%",minHeight:200,borderRadius:6,overflow:"hidden",border:"1px solid #2a2a2a"}}/>
    <VellumTooltip ai={ai} bins={bins} gs={gs} pinned={pinnedIdx!==null} topNSet={topNSet} showSize={false} showColor={false} showVol={showVol} ngMode="words" onDismiss={()=>onPin(null)}/>
    <div style={{position:"absolute",bottom:6,left:8,fontSize:9,color:"#333",fontFamily:"monospace"}}>{gs}² emotions</div>
  </div>)}

function EmoToggle({enabledSlots,setEnabledSlots}){
  const toggle=key=>{setEnabledSlots(prev=>{const n=new Set(prev);if(n.has(key))n.delete(key);else n.add(key);return n})};
  return(<div style={{display:"inline-grid",gridTemplateColumns:"repeat(3,10px)",gap:1}}>
    {EMO_LAYOUT.map(({r,c,emo},i)=>{const key=emo||"center";const on=enabledSlots.has(key);const col=emo?EC[emo]:"#ffffff";
      return(<div key={i} onClick={()=>toggle(key)} title={emo||"relevance"} style={{width:10,height:10,borderRadius:1,background:on?col:"#555",opacity:on?1:.5,cursor:"pointer",border:`1px solid ${on?col+"66":"#444"}`}}/>)})}</div>)}

function WordPanel({perDocData,selectedDocIds,filterWords,setFilterWords,sortBy,setSortBy,topN,ngMode,setNgMode}){
  const allActive=filterWords.size===0;
  const toggle=useCallback(w=>{setFilterWords(prev=>{const n=new Set(prev);if(n.has(w))n.delete(w);else n.add(w);return n})},[setFilterWords]);
  const unionWords=useMemo(()=>{const all=new Set();selectedDocIds.forEach(id=>{const r=perDocData[id];if(!r)return;if(ngMode==="words")r.topWords.forEach(([w])=>all.add(w));else if(ngMode==="bigrams")r.ng2.forEach(([w])=>all.add(w));else r.ng3.forEach(([w])=>all.add(w))});return[...all]},[perDocData,selectedDocIds,ngMode]);
  const getF=(w,id)=>{const r=perDocData[id];if(!r)return 0;if(ngMode==="words")return r.fMap[w]||0;if(ngMode==="bigrams")return r.ng2Map[w]||0;return r.ng3Map[w]||0};
  const getR=(w,id)=>{const r=perDocData[id];if(!r)return 0;if(ngMode==="words")return r.relevance[w]||0;return 0};
  const gMF=useMemo(()=>Math.max(...unionWords.map(w=>Math.max(...selectedDocIds.map(id=>getF(w,id)))),1),[unionWords,perDocData,selectedDocIds,ngMode]);
  const gMR=useMemo(()=>{if(ngMode!=="words")return 1;return Math.max(...unionWords.map(w=>Math.max(...selectedDocIds.map(id=>getR(w,id)))),1)},[unionWords,perDocData,selectedDocIds,ngMode]);
  const sorted=useMemo(()=>{const mx=w=>Math.max(...selectedDocIds.map(id=>sortBy==="relevance"&&ngMode==="words"?getR(w,id):getF(w,id)));return[...unionWords].sort((a,b)=>mx(b)-mx(a))},[unionWords,perDocData,selectedDocIds,sortBy,ngMode]);
  const sRB=ngMode==="words";
  return(<div style={{display:"flex",flexDirection:"column",gap:1,minWidth:0,height:"100%"}}>
    <div style={{display:"flex",gap:0,marginBottom:2,border:"1px solid #333",borderRadius:3,overflow:"hidden"}}>{[["words","1"],["bigrams","2"],["trigrams","3"]].map(([v,l])=>(<button key={v} onClick={()=>{setNgMode(v);setFilterWords(new Set())}} style={{flex:1,padding:"4px 0",background:ngMode===v?"#bb8fce":"#1a1a1a",color:ngMode===v?"#111":"#666",border:"none",cursor:"pointer",fontSize:10,fontFamily:"monospace",fontWeight:ngMode===v?"bold":"normal"}}>{l}</button>))}</div>
    <button onClick={()=>setFilterWords(new Set())} style={{padding:"4px 8px",marginBottom:2,background:allActive?"#4ecdc4":"#1a1a1a",color:allActive?"#111":"#999",border:`1px solid ${allActive?"#4ecdc4":"#333"}`,borderRadius:4,cursor:"pointer",fontSize:11,fontFamily:"monospace",fontWeight:allActive?"bold":"normal",textAlign:"left"}}>All</button>
    {sRB&&<div style={{display:"flex",gap:0,marginBottom:3,border:"1px solid #333",borderRadius:3,overflow:"hidden"}}>{[["freq","Freq"],["relevance","Relev"]].map(([v,l])=>(<button key={v} onClick={()=>setSortBy(v)} style={{flex:1,padding:"3px 0",background:sortBy===v?"#45b7d1":"#1a1a1a",color:sortBy===v?"#111":"#666",border:"none",cursor:"pointer",fontSize:9,fontFamily:"monospace"}}>{l}</button>))}</div>}
    <div style={{flex:1,overflowY:"auto",display:"flex",flexDirection:"column",gap:1}}>{sorted.map(w=>{const sel=filterWords.has(w),act=allActive||sel;const mF=Math.max(...selectedDocIds.map(id=>getF(w,id)));const mR=sRB?Math.max(...selectedDocIds.map(id=>getR(w,id))):0;const pr=mF>0;
      return(<button key={w} onClick={()=>toggle(w)} style={{padding:"3px 7px",background:sel?"#1a2a2a":"#151515",color:act&&pr?"#ddd":pr?"#777":"#444",border:`1px solid ${sel?"#4ecdc466":"#222"}`,borderRadius:3,cursor:"pointer",fontSize:10,fontFamily:"monospace",textAlign:"left",display:"flex",flexDirection:"column",gap:1,opacity:pr?1:.35}}>
        <div style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",width:"100%"}}>{w}</div>
        <div style={{display:"flex",alignItems:"center",gap:4,width:"100%",height:12}}><div style={{flex:1,height:3,background:"#222",borderRadius:1,overflow:"hidden"}}><div style={{width:`${(mF/gMF)*100}%`,height:"100%",background:act&&pr?"#4ecdc4":"#444",borderRadius:1,opacity:act?.8:.3}}/></div><span style={{width:24,textAlign:"right",fontSize:9,color:act&&pr?"#aaa":"#555",flexShrink:0}}>{mF||""}</span></div>
        {sRB&&<div style={{display:"flex",alignItems:"center",gap:4,width:"100%",height:12}}><div style={{flex:1,height:3,background:"#222",borderRadius:1,overflow:"hidden"}}><div style={{width:`${(mR/gMR)*100}%`,height:"100%",background:act&&pr?"#45b7d1":"#444",borderRadius:1,opacity:act?.8:.3}}/></div><span style={{width:24,textAlign:"right",fontSize:9,color:act&&pr?"#7ab8cc":"#444",flexShrink:0}}>{mR?mR.toFixed(0):""}</span></div>}
      </button>)})}</div>
    <div style={{display:"flex",gap:8,marginTop:3,fontSize:9,color:"#888"}}><span><span style={{color:"#4ecdc4"}}>—</span> freq</span>{sRB&&<span><span style={{color:"#45b7d1"}}>—</span> relev</span>}</div>
  </div>)}

function patchLayout(n){if(n<=1)return[1,1];if(n===2)return[1,2];if(n===3)return[1,3];if(n===4)return[2,2];if(n<=6)return[2,3];return[3,Math.ceil(n/3)]}

function Texturas(){
  const[docs,setDocs]=useState([{id:"d1",label:"Document 1",text:""}]);
  const[activeInputDoc,setActiveInputDoc]=useState("d1");
  const[selectedViewDocs,setSelectedViewDocs]=useState(new Set());
  const[tab,setTab]=useState("assets");
  const[topN,setTopN]=useState(25);const[wnDepth,setWnDepth]=useState(2);const[decay,setDecay]=useState(0.5);const[flow,setFlow]=useState("bi");
  const[gridSize,setGridSize]=useState(10);const[scale,setScale]=useState("log");
  const[showSize,setShowSize]=useState(true);const[showColor,setShowColor]=useState(false);const[showVol,setShowVol]=useState(false);
  const[filterWords,setFilterWords]=useState(new Set());const[sortBy,setSortBy]=useState("freq");const[ngMode,setNgMode]=useState("words");
  const[perDocResults,setPerDocResults]=useState({});
  const[loading,setLoading]=useState(false);const[msg,setMsg]=useState("");const[engSt,setEngSt]=useState(0);const[showParams,setShowParams]=useState(false);
  const[vellumPage,setVellumPage]=useState("channels");
  const[enabledSlots,setEnabledSlots]=useState(new Set([...EMOTIONS,"center"]));
  // Pin state lifted — persists across page toggles. Map of docId -> cellIdx
  const[pinnedCells,setPinnedCells]=useState({});
  const pinFor=useCallback((docId)=>(idx)=>setPinnedCells(prev=>({...prev,[docId]:idx})),[]);

  const eng=useRef({vec:new VectorEngine(),pos:new POSTagger(),lem:new Lemmatizer(),syn:new SynsetEngine(),sent:new SentimentScorer()});
  useEffect(()=>{if(!ASSET_BASE_URL)return;let c=false;
    (async()=>{const e=eng.current;
      const vj=await loadAsset("v-v","vectors/vocab.json",false,setMsg);const vb=await loadAsset("v-b","vectors/vectors.bin",true,setMsg);if(vj&&vb&&!c)await e.vec.loadBin(vj,vb);
      const pd=await loadAsset("w-p","wordnet/pos-lookup.json",false,setMsg);if(pd&&!c)e.pos.load(pd);
      const ld=await loadAsset("w-l","wordnet/lemmatizer.json",false,setMsg);if(ld&&!c)e.lem.load(ld);
      const sd=await loadAsset("w-s","wordnet/synsets.json",false,setMsg);if(sd&&!c)e.syn.load(sd);
      const el=await loadAsset("l-e","lexicons/nrc-emolex.json",false,setMsg);if(el&&!c)e.sent.lEl(el);
      const ni=await loadAsset("l-i","lexicons/nrc-intensity.json",false,setMsg);if(ni&&!c)e.sent.lInt(ni);
      const nv=await loadAsset("l-v","lexicons/nrc-vad.json",false,setMsg);if(nv&&!c)e.sent.lVad(nv);
      const va=await loadAsset("l-d","lexicons/vader.json",false,setMsg);if(va&&!c)e.sent.lVdr(va);
      const sw=await loadAsset("l-s","lexicons/sentiwordnet.json",false,setMsg);if(sw&&!c)e.sent.lSwn(sw);
      if(!c){setMsg("");setEngSt(s=>s+1)}})();return()=>{c=true}},[]);

  // Auto-rerun when assets arrive
  const hasRun=useRef(false);
  useEffect(()=>{if(Object.keys(perDocResults).length>0)hasRun.current=true},[perDocResults]);
  useEffect(()=>{if(engSt>0&&hasRun.current&&docs.filter(d=>d.text.trim()).length>0){
    const pdr={};docs.filter(d=>d.text.trim()).forEach(d=>{pdr[d.id]=analyzeForVellum(d.text,eng.current,topN,wnDepth,decay,flow)});
    setPerDocResults(pdr);setFilterWords(new Set());setNgMode("words")}},[engSt]);

  const addDoc=()=>{const id=`d${Date.now()}`;setDocs(d=>[...d,{id,label:`Document ${d.length+1}`,text:""}]);setActiveInputDoc(id)};
  const rmDoc=id=>{if(docs.length<=1)return;setDocs(d=>d.filter(x=>x.id!==id));if(activeInputDoc===id)setActiveInputDoc(docs[0]?.id);setSelectedViewDocs(prev=>{const n=new Set(prev);n.delete(id);return n})};
  const updDoc=(id,f,v)=>setDocs(d=>d.map(x=>x.id===id?{...x,[f]:v}:x));
  const handleFiles=files=>{Array.from(files).forEach(f=>{if(!f.name.endsWith(".txt"))return;const id=`d${Date.now()}_${Math.random().toString(36).slice(2,6)}`;const reader=new FileReader();reader.onload=ev=>{setDocs(d=>[...d.filter(x=>x.text.trim()),{id,label:f.name.replace(".txt",""),text:ev.target.result}])};reader.readAsText(f)})};
  const parseSeparators=text=>{const parts=text.split(/---DOC(?::?\s*([^-]*))?\s*---/i);const result=[];let pl=null;for(let i=0;i<parts.length;i++){const t=parts[i]?.trim();if(!t)continue;if(i%2===1)pl=t;else{result.push({id:`d${Date.now()}_${i}`,label:pl||`Document ${result.length+1}`,text:t});pl=null}}return result.length>1?result:null};

  const validDocs=docs.filter(d=>d.text.trim());const analyzedIds=Object.keys(perDocResults);
  const selectedArr=[...selectedViewDocs].filter(id=>perDocResults[id]);const isPatchwork=selectedArr.length>1;

  const runAnalysis=useCallback(()=>{if(!validDocs.length)return;setLoading(true);setMsg("Analyzing...");
    setTimeout(()=>{const pdr={};validDocs.forEach(d=>{pdr[d.id]=analyzeForVellum(d.text,eng.current,topN,wnDepth,decay,flow)});
      setPerDocResults(pdr);setFilterWords(new Set());setNgMode("words");setSelectedViewDocs(new Set([validDocs[0].id]));setPinnedCells({});setLoading(false);setMsg("");setTab("vellum")},50)},[docs,topN,wnDepth,decay,flow]);

  const handleDocClick=(id,e)=>{if(e.ctrlKey||e.metaKey){setSelectedViewDocs(prev=>{const n=new Set(prev);if(n.has(id))n.delete(id);else n.add(id);if(n.size===0)n.add(id);return n})}else setSelectedViewDocs(new Set([id]))};
  const rerunFlow=v=>{setFlow(v);setTimeout(()=>{const pdr={};validDocs.forEach(d=>{pdr[d.id]=analyzeForVellum(d.text,eng.current,topN,wnDepth,decay,v)});setPerDocResults(pdr);setFilterWords(new Set())},50)};
  const rerunTopN=useCallback(n=>{setTopN(n);if(!validDocs.length)return;setTimeout(()=>{const pdr={};validDocs.forEach(d=>{pdr[d.id]=analyzeForVellum(d.text,eng.current,n,wnDepth,decay,flow)});setPerDocResults(pdr);setFilterWords(new Set());setNgMode("words")},50)},[docs,wnDepth,decay,flow]);

  const allVData=useMemo(()=>{const out={};selectedArr.forEach(id=>{const r=perDocResults[id];if(!r)return;const ngFM=ngMode==="bigrams"?r.ng2Map:ngMode==="trigrams"?r.ng3Map:null;out[id]=vellumBins(r.enriched,gridSize,filterWords,ngMode,ngFM)});return out},[perDocResults,selectedArr.join(","),gridSize,filterWords,ngMode]);
  const normMaxes=useMemo(()=>{let mR=0,mV=0,mA=0;Object.values(allVData).forEach(vd=>{vd.bins.forEach(b=>{if(!b.empty&&!b.dimmed){mR=Math.max(mR,Math.abs(b.rel));mV=Math.max(mV,Math.abs(b.vader));mA=Math.max(mA,Math.abs(b.arousal))}})});return{rel:mR||1,vader:mV||.01,arousal:mA||1}},[allVData]);

  const patchContainerRef=useRef();const[patchCellW,setPatchCellW]=useState(0);
  useEffect(()=>{if(!isPatchwork||!patchContainerRef.current)return;const ro=new ResizeObserver(en=>{const{width}=en[0].contentRect;const[,cols]=patchLayout(selectedArr.length);const cw=Math.floor((width-6*(cols-1))/cols);if(cw>10)setPatchCellW(cw)});ro.observe(patchContainerRef.current);return()=>ro.disconnect()},[isPatchwork,selectedArr.length]);

  const e=eng.current;const aC=(showSize?1:0)+(showColor?1:0)+(showVol?1:0);
  const tS=()=>{if(showSize&&aC<=1)return;setShowSize(!showSize)};const tC=()=>{if(showColor&&aC<=1)return;setShowColor(!showColor)};const tV=()=>{if(showVol&&aC<=1)return;setShowVol(!showVol)};

  const mainTabs=[{id:"assets",l:"Assets"},{id:"input",l:"Input"},{id:"vellum",l:"Vellum"}];
  const[rows,cols]=patchLayout(selectedArr.length);const gridH=isPatchwork?Math.max(240,Math.min(380,520/rows)):540;

  return(<div style={{background:"#111",color:"#ddd",minHeight:"100vh",fontFamily:"monospace",display:"flex",flexDirection:"column"}}>
    <div style={{padding:"12px 20px",borderBottom:"1px solid #2a2a2a",display:"flex",alignItems:"center",gap:12}}>
      <span style={{fontSize:18,color:"#4ecdc4",fontWeight:"bold"}}>⬡ Vellum</span><span style={{fontSize:11,color:"#555"}}>Texturas v0.7</span>
      {analyzedIds.length>0&&<span style={{fontSize:10,color:"#ccc",marginLeft:8}}>● {analyzedIds.length} doc{analyzedIds.length>1?"s":""}</span>}
      <div style={{marginLeft:"auto",display:"flex",gap:8}}>{e.vec.isLoaded()&&<span style={{fontSize:10,color:"#82e0aa"}}>● vec</span>}{e.sent.ready&&<span style={{fontSize:10,color:"#f7dc6f"}}>● sent</span>}{e.pos.ready&&<span style={{fontSize:10,color:"#bb8fce"}}>● nlp</span>}{e.syn.ready&&<span style={{fontSize:10,color:"#45b7d1"}}>● wn</span>}</div></div>
    <div style={{display:"flex",borderBottom:"1px solid #2a2a2a"}}>{mainTabs.map(t=>(<button key={t.id} onClick={()=>setTab(t.id)} style={{padding:"10px 14px",background:tab===t.id?"#1a1a1a":"transparent",color:tab===t.id?"#4ecdc4":"#888",border:"none",borderBottom:tab===t.id?"2px solid #4ecdc4":"2px solid transparent",cursor:"pointer",fontSize:12,fontFamily:"monospace"}}>{t.l}</button>))}</div>
    {tab==="vellum"&&analyzedIds.length>0&&(<div style={{display:"flex",gap:4,padding:"8px 20px",borderBottom:"1px solid #2a2a2a",background:"#151515",overflowX:"auto",alignItems:"center"}}>
      {analyzedIds.length>1&&<span style={{fontSize:9,color:"#555",marginRight:4}}>Ctrl+click patchwork</span>}
      {validDocs.filter(d=>perDocResults[d.id]).map(d=>{const sel=selectedViewDocs.has(d.id);const tw=perDocResults[d.id]?.tw||0;
        return(<button key={d.id} onClick={ev=>handleDocClick(d.id,ev)} style={{padding:"4px 12px",borderRadius:3,border:`1px solid ${sel?"#45b7d1":"#333"}`,background:sel?"#45b7d1":"#1a1a1a",color:sel?"#111":"#888",fontSize:11,fontFamily:"monospace",cursor:"pointer"}}>{d.label} <span style={{opacity:.6,fontSize:9}}>({tw.toLocaleString()})</span></button>)})}
    </div>)}
    <div style={{flex:1,padding:"16px 20px",overflowY:"auto"}}>
      {tab==="assets"&&(<div style={{maxWidth:560,margin:"0 auto"}}><h3 style={{color:"#4ecdc4",fontSize:15,marginBottom:16,fontWeight:"normal"}}>Asset Status</h3>
        {[["GloVe",e.vec.isLoaded()?`${e.vec.vocab.toLocaleString()} × ${e.vec.dim}d`:null],["POS",e.pos.ready?"ready":null],["Lemmatizer",e.lem.ready?"ready":null],["Synsets",e.syn.ready?"ready":null],
          ["NRC EmoLex",e.sent.el?`${Object.keys(e.sent.el).length.toLocaleString()}`:null],["NRC Intensity",e.sent.int?`${Object.keys(e.sent.int).length.toLocaleString()}`:null],
          ["NRC VAD",e.sent.vad?`${Object.keys(e.sent.vad).length.toLocaleString()}`:null],["VADER",e.sent.vdr?`${Object.keys(e.sent.vdr).length.toLocaleString()}`:null],
          ["SentiWordNet",e.sent.swn?`${Object.keys(e.sent.swn).length.toLocaleString()}`:null]
        ].map(([l,s])=>(<div key={l} style={{display:"flex",alignItems:"center",gap:10,padding:"8px 12px",background:"#1a1a1a",borderRadius:4,marginBottom:4,border:`1px solid ${s?"#333":"#2a2a2a"}`}}><span style={{fontSize:14}}>{s?"✓":"○"}</span><span style={{fontSize:12,color:s?"#ccc":"#666",flex:1}}>{l}</span><span style={{fontSize:11,color:s?"#82e0aa":"#444"}}>{s||"–"}</span></div>))}</div>)}
      {tab==="input"&&(<div style={{maxWidth:800,margin:"0 auto"}}>
        <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}><span style={{fontSize:13,color:"#aaa"}}>Documents ({docs.length})</span>
          <button onClick={addDoc} style={{padding:"4px 12px",background:"#2a2a2a",color:"#4ecdc4",border:"1px solid #444",borderRadius:4,fontSize:12,fontFamily:"monospace",cursor:"pointer"}}>+ Add</button>
          <label style={{padding:"4px 12px",background:"#2a2a2a",color:"#45b7d1",border:"1px solid #444",borderRadius:4,fontSize:12,fontFamily:"monospace",cursor:"pointer"}}>Upload .txt<input type="file" multiple accept=".txt" style={{display:"none"}} onChange={ev=>handleFiles(ev.target.files)}/></label></div>
        <div style={{display:"flex",gap:4,marginBottom:12,flexWrap:"wrap"}}>{docs.map(d=>(<button key={d.id} onClick={()=>setActiveInputDoc(d.id)} style={{padding:"5px 12px",borderRadius:4,border:`1px solid ${activeInputDoc===d.id?"#45b7d1":"#333"}`,background:activeInputDoc===d.id?"#1a2a2a":"#1a1a1a",color:activeInputDoc===d.id?"#45b7d1":"#888",fontSize:12,fontFamily:"monospace",cursor:"pointer"}}>{d.label}{docs.length>1&&<span onClick={ev=>{ev.stopPropagation();rmDoc(d.id)}} style={{marginLeft:8,color:"#666",cursor:"pointer"}}>×</span>}</button>))}</div>
        {docs.filter(d=>d.id===activeInputDoc).map(d=>{const hm=d.text.includes("---DOC");return(<div key={d.id}>
          <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:8}}><input type="text" value={d.label} onChange={ev=>updDoc(d.id,"label",ev.target.value)} style={{width:300,padding:"6px 10px",background:"#1a1a1a",border:"1px solid #444",borderRadius:4,color:"#ccc",fontSize:13,fontFamily:"monospace",boxSizing:"border-box"}}/>
            {hm&&<button onClick={()=>{const p=parseSeparators(d.text);if(p){setDocs(p);setActiveInputDoc(p[0].id)}}} style={{padding:"6px 14px",background:"#2a2a1a",color:"#f0b27a",border:"1px solid #f0b27a44",borderRadius:4,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>Split ({(d.text.match(/---DOC/gi)||[]).length})</button>}</div>
          <textarea value={d.text} onChange={ev=>updDoc(d.id,"text",ev.target.value)} onPaste={ev=>{const t=ev.clipboardData.getData("text");if(t.includes("---DOC")){ev.preventDefault();const p=parseSeparators(t);if(p){setDocs(p);setActiveInputDoc(p[0].id)}}}}
            placeholder="Paste text here...\n\nUse ---DOC: Label--- separators for multiple documents" style={{width:"100%",height:"calc(100vh - 400px)",background:"#0d0d0d",border:"1px solid #2a2a2a",borderRadius:6,color:"#ccc",padding:16,fontSize:13,fontFamily:"monospace",resize:"none",boxSizing:"border-box",lineHeight:1.6}}/></div>)})}
        <div style={{marginTop:12,display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}}>
          <button onClick={()=>setShowParams(!showParams)} style={{padding:"6px 12px",background:"#1a1a1a",color:"#888",border:"1px solid #333",borderRadius:4,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>Parameters {showParams?"▾":"▸"}</button>
          <button onClick={runAnalysis} disabled={!validDocs.length||loading} style={{padding:"8px 20px",background:validDocs.length&&!loading?"#4ecdc4":"#333",color:validDocs.length&&!loading?"#111":"#666",border:"none",borderRadius:4,cursor:validDocs.length&&!loading?"pointer":"default",fontFamily:"monospace",fontSize:13,fontWeight:"bold"}}>Analyze {validDocs.length} doc{validDocs.length!==1?"s":""} →</button>
          {msg&&<span style={{fontSize:10,color:"#f7dc6f"}}>{msg}</span>}</div>
        {showParams&&(<div style={{marginTop:10,padding:14,background:"#1a1a1a",borderRadius:6,border:"1px solid #333",display:"flex",gap:20,flexWrap:"wrap",alignItems:"flex-end"}}>
          <div><label style={{fontSize:10,color:"#888",display:"block",marginBottom:3}}>Top N: {topN}</label><input type="range" min={10} max={50} value={topN} onChange={ev=>setTopN(+ev.target.value)} style={{width:100}}/></div>
          <div><label style={{fontSize:10,color:"#888",display:"block",marginBottom:3}}>WN depth: {wnDepth}</label><input type="range" min={1} max={3} value={wnDepth} onChange={ev=>setWnDepth(+ev.target.value)} style={{width:80}}/></div>
          <div><label style={{fontSize:10,color:"#888",display:"block",marginBottom:3}}>Decay: {decay.toFixed(2)}</label><input type="range" min={30} max={80} value={decay*100} onChange={ev=>setDecay(+ev.target.value/100)} style={{width:100}}/></div>
          <div><label style={{fontSize:10,color:"#888",display:"block",marginBottom:3}}>Flow:</label><div style={{display:"flex",gap:0,border:"1px solid #333",borderRadius:3,overflow:"hidden"}}>{[["bi","Bi"],["up","Up ↑"],["down","Dn ↓"]].map(([v,l])=>(<button key={v} onClick={()=>setFlow(v)} style={{padding:"4px 8px",background:flow===v?"#45b7d1":"#1a1a1a",color:flow===v?"#111":"#666",border:"none",cursor:"pointer",fontSize:10,fontFamily:"monospace"}}>{l}</button>))}</div></div>
        </div>)}</div>)}
      {tab==="vellum"&&selectedArr.length>0&&(<div style={{maxWidth:1100,margin:"0 auto"}}>
        <div style={{display:"flex",gap:8,marginBottom:12,alignItems:"center",height:36}}>
          <div style={{display:"flex",gap:0,border:"1px solid #333",borderRadius:4,overflow:"hidden"}}>
            <button onClick={()=>setVellumPage("channels")} style={{padding:"5px 11px",background:vellumPage==="channels"?"#4ecdc4":"#1a1a1a",color:vellumPage==="channels"?"#111":"#666",border:"none",cursor:"pointer",fontSize:11,fontFamily:"monospace",fontWeight:vellumPage==="channels"?"bold":"normal"}}>◉ Channels</button>
            <button onClick={()=>setVellumPage("emotions")} style={{padding:"5px 11px",background:vellumPage==="emotions"?"#f0b27a":"#1a1a1a",color:vellumPage==="emotions"?"#111":"#666",border:"none",cursor:"pointer",fontSize:11,fontFamily:"monospace",fontWeight:vellumPage==="emotions"?"bold":"normal"}}>◉ Emotions</button></div>
          <div style={{width:1,height:22,background:"#333"}}/>
          <div style={{display:"flex",gap:0,border:"1px solid #333",borderRadius:4,overflow:"hidden"}}>{[10,20,30].map(g=>(<button key={g} onClick={()=>setGridSize(g)} style={{padding:"5px 11px",background:gridSize===g?"#bb8fce":"#1a1a1a",color:gridSize===g?"#111":"#666",border:"none",cursor:"pointer",fontSize:11,fontFamily:"monospace",fontWeight:gridSize===g?"bold":"normal"}}>{g}²</button>))}</div>
          <div style={{width:1,height:22,background:"#333"}}/>
          <div style={{display:"flex",gap:0,border:"1px solid #333",borderRadius:4,overflow:"hidden"}}>{["linear","log"].map(m=>(<button key={m} onClick={()=>setScale(m)} style={{padding:"5px 11px",background:scale===m?"#4ecdc4":"#1a1a1a",color:scale===m?"#111":"#666",border:"none",cursor:"pointer",fontSize:11,fontFamily:"monospace",fontWeight:scale===m?"bold":"normal"}}>{m}</button>))}</div>
          <div style={{width:1,height:22,background:"#333"}}/>
          <div style={{display:"flex",alignItems:"center",gap:5,width:145,flexShrink:0,justifyContent:vellumPage==="channels"?"flex-start":"flex-end"}}>
            {vellumPage==="channels"&&<button onClick={tS} title="Semantic relevance" style={{padding:"5px 10px",background:showSize?"#0d2a28":"#1a1a1a",color:showSize?"#4ecdc4":"#555",border:`1px solid ${showSize?"#4ecdc444":"#333"}`,borderRadius:4,cursor:showSize&&aC<=1?"not-allowed":"pointer",fontSize:11,fontFamily:"monospace",opacity:showSize&&aC<=1?.5:1}}>Freq/Rel</button>}
            {vellumPage==="channels"?<button onClick={tC} title="VADER polarity" style={{padding:"5px 10px",background:showColor?"#1a2a2a":"#1a1a1a",color:showColor?"#82e0aa":"#555",border:`1px solid ${showColor?"#82e0aa44":"#333"}`,borderRadius:4,cursor:showColor&&aC<=1?"not-allowed":"pointer",fontSize:11,fontFamily:"monospace",opacity:showColor&&aC<=1?.5:1}}>Emotion</button>:<EmoToggle enabledSlots={enabledSlots} setEnabledSlots={setEnabledSlots}/>}
          </div>
          <button onClick={vellumPage==="channels"?tV:()=>setShowVol(!showVol)} title="NRC VAD Arousal" style={{padding:"5px 10px",background:showVol?"#2a2a1a":"#1a1a1a",color:showVol?"#f7dc6f":"#555",border:`1px solid ${showVol?"#f7dc6f44":"#333"}`,borderRadius:4,cursor:vellumPage==="channels"&&showVol&&aC<=1?"not-allowed":"pointer",fontSize:11,fontFamily:"monospace",opacity:vellumPage==="channels"&&showVol&&aC<=1?.5:1}}>Arousal</button>
          <div style={{width:1,height:22,background:"#333"}}/>
          <div style={{display:"flex",gap:0,border:"1px solid #333",borderRadius:4,overflow:"hidden"}}>
            {[["bi","Bi","Bidirectional"],["up","↑","Upward"],["down","↓","Downward"]].map(([v,l,tip])=>(<button key={v} onClick={()=>rerunFlow(v)} title={tip} style={{padding:"5px 9px",background:flow===v?"#45b7d1":"#1a1a1a",color:flow===v?"#111":"#666",border:"none",cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>{l}</button>))}</div>
          <div style={{width:1,height:22,background:"#333"}}/>
          <div style={{display:"flex",alignItems:"center",gap:5}}><span style={{fontSize:10,color:"#666"}}>N:</span><input type="range" min={10} max={50} value={topN} onChange={ev=>rerunTopN(+ev.target.value)} style={{width:60}}/><span style={{fontSize:10,color:"#aaa",width:16}}>{topN}</span></div>
          {isPatchwork&&<span style={{fontSize:10,color:"#f0b27a",marginLeft:4}}>patchwork</span>}
        </div>
        <div style={{display:"flex",gap:10,alignItems:"stretch"}}>
          <div style={{flex:1,minWidth:0,overflow:"hidden"}}>
            {isPatchwork?(<div ref={patchContainerRef} style={{display:"grid",gridTemplateColumns:`repeat(${cols},1fr)`,gap:6}}>
              {selectedArr.map(id=>{const vd=allVData[id];const doc=docs.find(d=>d.id===id);if(!vd)return null;
                return(<div key={`pw-${id}-${vellumPage}`} style={{height:gridH,minWidth:0}}>
                  {vellumPage==="channels"?<VellumGrid key={`vc-${id}-${gridSize}-${patchCellW}`} bins={vd.bins} scale={scale} showSize={showSize} showColor={showColor} showVol={showVol} gridSize={gridSize} normMaxes={normMaxes} label={doc?.label} ngMode={ngMode} fixedWidth={patchCellW||undefined} fixedHeight={gridH} topNWords={perDocResults[id]?.topWords} pinnedIdx={pinnedCells[id]??null} onPin={pinFor(id)}/>
                  :<VellumEmoGrid key={`ve-${id}-${gridSize}-${patchCellW}`} bins={vd.bins} scale={scale} showVol={showVol} gridSize={gridSize} normMaxes={normMaxes} label={doc?.label} fixedWidth={patchCellW||undefined} fixedHeight={gridH} enabledSlots={enabledSlots} topNWords={perDocResults[id]?.topWords} pinnedIdx={pinnedCells[id]??null} onPin={pinFor(id)}/>}
                </div>)})}
            </div>):(<div style={{height:540}}>
              {allVData[selectedArr[0]]&&(vellumPage==="channels"
                ?<VellumGrid key={`vc-${selectedArr[0]}-${gridSize}`} bins={allVData[selectedArr[0]].bins} scale={scale} showSize={showSize} showColor={showColor} showVol={showVol} gridSize={gridSize} normMaxes={normMaxes} label={analyzedIds.length>1?docs.find(d=>d.id===selectedArr[0])?.label:null} ngMode={ngMode} topNWords={perDocResults[selectedArr[0]]?.topWords} pinnedIdx={pinnedCells[selectedArr[0]]??null} onPin={pinFor(selectedArr[0])}/>
                :<VellumEmoGrid key={`ve-${selectedArr[0]}-${gridSize}`} bins={allVData[selectedArr[0]].bins} scale={scale} showVol={showVol} gridSize={gridSize} normMaxes={normMaxes} label={analyzedIds.length>1?docs.find(d=>d.id===selectedArr[0])?.label:null} enabledSlots={enabledSlots} topNWords={perDocResults[selectedArr[0]]?.topWords} pinnedIdx={pinnedCells[selectedArr[0]]??null} onPin={pinFor(selectedArr[0])}/>)}
            </div>)}
          </div>
          <div style={{width:160,flexShrink:0,maxHeight:isPatchwork?gridH*rows+6*(rows-1):540,display:"flex",flexDirection:"column"}}>
            <div style={{fontSize:10,color:"#888",marginBottom:3}}>Top {topN} · click to filter</div>
            <WordPanel perDocData={perDocResults} selectedDocIds={selectedArr} filterWords={filterWords} setFilterWords={setFilterWords} sortBy={sortBy} setSortBy={setSortBy} topN={topN} ngMode={ngMode} setNgMode={setNgMode}/>
          </div>
        </div>
        <div style={{display:"flex",gap:14,marginTop:10,padding:"8px 12px",background:"#0d0d0d",borderRadius:4,border:"1px solid #1a1a1a",fontSize:11,color:"#555",flexWrap:"wrap"}}>
          <span>View: <span style={{color:vellumPage==="channels"?"#4ecdc4":"#f0b27a"}}>{vellumPage}</span></span>
          <span>Grid: <span style={{color:"#bb8fce"}}>{gridSize}²</span></span>
          <span>Flow: <span style={{color:"#45b7d1"}}>{flow==="bi"?"bi":flow==="up"?"↑":"↓"}</span></span>
          {filterWords.size>0&&<span>Filter: <span style={{color:"#f0b27a"}}>{filterWords.size}</span></span>}
          {isPatchwork&&<span>Patchwork: <span style={{color:"#f0b27a"}}>{selectedArr.length}</span></span>}
        </div>
        <div style={{display:"flex",gap:16,marginTop:6,padding:"6px 12px",background:"#0d0d0d",borderRadius:4,border:"1px solid #1a1a1a",fontSize:10,color:"#555",flexWrap:"wrap"}}>
          {vellumPage==="channels"?(<>
            <span title="Semantic relevance" style={{cursor:"help",borderBottom:"1px dotted #444"}}><span style={{color:showSize?"#4ecdc4":"#666"}}>■</span> Freq/Rel: semantic gravity</span>
            <span title="VADER polarity" style={{cursor:"help",borderBottom:"1px dotted #444"}}><span style={{color:showColor?"#82e0aa":"#666"}}>■</span>/<span style={{color:showColor?"#ff6b6b":"#666"}}>■</span> Emotion: VADER polarity</span>
            <span title="NRC VAD Arousal" style={{cursor:"help",borderBottom:"1px dotted #444"}}><span style={{color:showVol?"#f7dc6f":"#666"}}>▮</span> Arousal: emotional intensity</span>
          </>):(<>
            <span title="Plutchik 8" style={{cursor:"help",borderBottom:"1px dotted #444",color:"#f0b27a"}}>Plutchik 8 emotions</span>
            <span style={{cursor:"help",borderBottom:"1px dotted #444"}}>brightness = emotion proportion</span>
            <span style={{cursor:"help",borderBottom:"1px dotted #444"}}>center = relevance</span>
            {showVol&&<span style={{cursor:"help",borderBottom:"1px dotted #444",color:"#f7dc6f"}}>height = per-emotion arousal</span>}
          </>)}
        </div>
      </div>)}
      {tab==="vellum"&&selectedArr.length===0&&(<div style={{color:"#555",textAlign:"center",marginTop:80,fontSize:13}}>← Add documents in Input tab, then click Analyze.</div>)}
    </div>
  </div>)}
