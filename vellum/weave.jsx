const { useState, useRef, useEffect, useCallback, useMemo } = React;

// ═══ CONFIG ═══
const ASSET_BASE_URL="https://epsidebox.github.io/texturas/assets/";
const STOP_WORDS=new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","shall","should","may","might","can","could","this","that","these","those","it","its","i","me","my","mine","we","us","our","ours","you","your","yours","he","him","his","she","her","hers","they","them","their","theirs","what","which","who","whom","whose","where","when","how","why","if","then","else","so","because","as","than","too","very","just","about","above","after","again","all","also","am","any","before","below","between","both","during","each","few","further","get","got","here","into","more","most","much","only","other","out","over","own","same","some","such","there","through","under","until","up","while","down","like","make","many","now","off","once","one","onto","per","since","still","take","thing","things","think","two","upon","well","went","yeah","yes","oh","um","uh","ah","okay","ok","really","actually","basically","literally","right","know","going","go","come","say","said","tell","told","see","look","want","need","way","back","even","around","something","anything","everything","someone","anyone","everyone","always","sometimes","often","already","yet","kind","sort","lot","bit","little","big","good","great","long","new","old","first","last","next","time","times","day","days","year","years","people","person","part","place","point","work","life","world","hand","week","end","case","fact","area","number","given","able","made","set","used","using","another","different","example","enough","however","important","include","including","keep","large","whether","without","within","along","become","across","among","toward","towards","though","although","either","rather","several","certain","less","likely","began","begun","brought","thus","therefore","hence","moreover","furthermore","nevertheless","nonetheless","meanwhile","accordingly","consequently","subsequently"]);
const NEGATION=new Set(["not","no","never","neither","nor","don't","doesn't","didn't","won't","wouldn't","can't","couldn't","shouldn't","isn't","aren't","wasn't","weren't","haven't","hasn't","hadn't","cannot","nothing","nobody","none","nowhere"]);
const EMOTIONS=["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"];
const EC={anger:"#ff6b6b",anticipation:"#f7dc6f",disgust:"#bb8fce",fear:"#85c1e9",joy:"#82e0aa",sadness:"#45b7d1",surprise:"#f0b27a",trust:"#4ecdc4"};
const CC=["#ff6b6b","#4ecdc4","#45b7d1","#f7dc6f","#bb8fce","#82e0aa","#f0b27a","#85c1e9","#f1948a","#73c6b6"];
const LAYER_CFG=[
  {id:"polarity",label:"Polarity",color:"#82e0aa",desc:"VADER → text color"},
  {id:"emotion",label:"Emotion",color:"#f0b27a",desc:"Plutchik → underline colors"},
  {id:"frequency",label:"Frequency",color:"#bb8fce",desc:"Corpus freq → brightness"},
  {id:"relevance",label:"Relevance",color:"#4ecdc4",desc:"Activation → font weight"},
  {id:"community",label:"Community",color:"#45b7d1",desc:"Louvain → background"}
];
const EMO_LAYOUT=[{r:0,c:0,emo:"anticipation"},{r:0,c:1,emo:"joy"},{r:0,c:2,emo:"trust"},{r:1,c:0,emo:"anger"},{r:1,c:1,emo:null},{r:1,c:2,emo:"fear"},{r:2,c:0,emo:"disgust"},{r:2,c:1,emo:"sadness"},{r:2,c:2,emo:"surprise"}];

// ═══ NLP ENGINES ═══
class POSTagger{constructor(){this.lk=null;this.ready=false}load(d){this.lk=d;this.ready=true}
sfx=[["tion","n"],["sion","n"],["ment","n"],["ness","n"],["ity","n"],["ance","n"],["ence","n"],["ism","n"],["ist","n"],["ting","v"],["sing","v"],["ning","v"],["ing","v"],["ated","v"],["ized","v"],["ed","v"],["ize","v"],["ify","v"],["ously","r"],["ively","r"],["fully","r"],["ally","r"],["ly","r"],["ful","a"],["ous","a"],["ive","a"],["able","a"],["ible","a"],["ical","a"],["less","a"],["al","a"]];
tag(w){const l=w.toLowerCase();if(this.lk?.[l])return this.lk[l];for(const[s,p]of this.sfx)if(l.endsWith(s))return p;return"n"}}

class Lemmatizer{constructor(){this.exc=null;this.rl=null;this.ready=false}load(d){this.exc=d.exceptions;this.rl=d.rules;this.ready=true}
lemmatize(w,pos){const l=w.toLowerCase();if(this.exc?.[pos]?.[l])return this.exc[pos][l];for(const[s,r]of(this.rl?.[pos]||[]))if(l.endsWith(s)&&l.length>s.length)return l.slice(0,-s.length)+r;return l}}

class SynsetEngine{constructor(){this.d=null;this.ready=false}load(d){this.d=d;this.ready=true}
getRels(w,p){const e=this.d?.[w+"#"+p];if(!e)return{h:[],y:[],m:[]};return{h:e.hypernyms||[],y:e.hyponyms||[],m:e.meronyms||[]}}
directed(w,p,flow){const r=this.getRels(w,p);if(flow==="up")return[...new Set([...r.h,...r.m])];if(flow==="down")return[...new Set([...r.y,...r.m])];return[...new Set([...r.h,...r.y,...r.m])]}}

class SentimentScorer{constructor(){this.el=null;this.int=null;this.vad=null;this.vdr=null;this.swn=null;this.ready=false}
lEl(d){this.el=d;this._c()}lInt(d){this.int=d;this._c()}lVad(d){this.vad=d;this._c()}lVdr(d){this.vdr=d;this._c()}lSwn(d){this.swn=d;this._c()}
_c(){this.ready=!!(this.el||this.int||this.vad||this.vdr||this.swn)}
score(lem,pos){const r={lemma:lem,pos};if(this.el?.[lem])r.emolex=this.el[lem];if(this.int?.[lem])r.intensity=this.int[lem];if(this.vad?.[lem])r.vad=this.vad[lem];
if(this.vdr?.[lem]!==undefined)r.vader=this.vdr[lem];const k=lem+"#"+pos;if(this.swn?.[k])r.swn=this.swn[k];
else{const vs=["n","v","a","r"].map(p=>this.swn?.[lem+"#"+p]).filter(Boolean);if(vs.length)r.swn=vs[0]}
r.has=!!(r.emolex||r.intensity||r.vad||r.vader!==undefined||r.swn);return r}}

// ═══ SPREADING ACTIVATION ═══
function spreadAct(fMap,syn,pos,depth,decay,flow){
  if(!syn.ready)return{...fMap};const corpus=new Set(Object.keys(fMap)),scores={};
  for(const l of corpus)scores[l]=fMap[l]||0;
  for(const[l,f]of Object.entries(fMap)){if(f===0)continue;const p=pos.ready?pos.tag(l):"n";
    let front=[{w:l,p}],vis=new Set([l]);
    for(let h=1;h<=depth;h++){const nf=[];for(const nd of front){const tgts=syn.directed(nd.w,nd.p,flow);
      for(const t of tgts){if(vis.has(t))continue;vis.add(t);const amt=f*Math.pow(decay,h);
        if(corpus.has(t))scores[t]=(scores[t]||0)+amt;nf.push({w:t,p:pos.ready?pos.tag(t):"n"})}}front=nf}}
  return scores}

// ═══ ANALYSIS ═══
const getNg=(ts,n)=>{const g=[];for(let i=0;i<=ts.length-n;i++)g.push(ts.slice(i,i+n).join(" "));return _.chain(g).countBy().toPairs().sortBy(1).reverse().value()};

function analyzeForWeave(text,eng,topN,winSize,wnDepth,decay,flow){
  const norm=text.replace(/[\u2018\u2019\u201A\u201B]/g,"'").replace(/[\u201C\u201D\u201E\u201F]/g,'"');
  const paras=norm.split(/\n\s*\n+/).filter(p=>p.trim());
  const allWords=[];
  const renderParas=paras.map(para=>{const parts=[];const rx=/([\w'-]+)|(\s+)|([^\w\s'-]+)/g;let m;
    while((m=rx.exec(para))!==null){if(m[1]){const surface=m[1],lower=surface.toLowerCase();
      const isStop=STOP_WORDS.has(lower)&&!NEGATION.has(lower);const p=eng.pos.ready?eng.pos.tag(lower):"n";
      const lemma=eng.lem.ready?eng.lem.lemmatize(lower,p):lower;
      if(!isStop)allWords.push(lemma);parts.push({type:"word",surface,lower,lemma,pos:p,isStop})}
    else parts.push({type:"other",surface:m[0]})}return parts});
  const freqMap=_.countBy(allWords),freqPairs=_.chain(freqMap).toPairs().sortBy(1).reverse().value();
  const topWords=freqPairs.slice(0,topN).map(f=>f[0]),maxFreq=freqPairs[0]?.[1]||1;
  const ng2=getNg(allWords,2).slice(0,topN),ng3=getNg(allWords,3).slice(0,topN);
  // Co-occurrence + Louvain
  const ws=new Set(topWords),mx={};topWords.forEach(a=>{mx[a]={};topWords.forEach(b=>{mx[a][b]=0})});
  for(let i=0;i<allWords.length;i++){if(!ws.has(allWords[i]))continue;for(let j=Math.max(0,i-winSize);j<=Math.min(allWords.length-1,i+winSize);j++){if(i!==j&&ws.has(allWords[j]))mx[allWords[i]][allWords[j]]++}}
  const n=topWords.length;let cm=topWords.map((_,i)=>i),tw2=0;const wt={};
  topWords.forEach(a=>{wt[a]=0;topWords.forEach(b=>{const w=mx[a]?.[b]||0;tw2+=w;wt[a]+=w})});tw2/=2;
  if(tw2>0){let imp=true,it=0;while(imp&&it<20){imp=false;it++;for(let i=0;i<n;i++){let bc=cm[i],bg=0;
    for(const c of new Set(cm)){if(c===cm[i])continue;let g=0;for(let j=0;j<n;j++){if(cm[j]!==c)continue;g+=(mx[topWords[i]]?.[topWords[j]]||0)-(wt[topWords[i]]*wt[topWords[j]])/(2*tw2)}if(g>bg){bg=g;bc=c}}
    if(bc!==cm[i]){cm[i]=bc;imp=true}}}}
  const uComms=[...new Set(cm)],commMap={};topWords.forEach((w,i)=>{commMap[w]=uComms.indexOf(cm[i])});
  const relevanceMap=spreadAct(freqMap,eng.syn,eng.pos,wnDepth,decay,flow);
  const maxRel=Math.max(...Object.values(relevanceMap),1);
  const sentCache={};function getSent(lem,p){if(sentCache[lem])return sentCache[lem];const s=eng.sent.ready?eng.sent.score(lem,p):{};sentCache[lem]=s;return s}
  const enriched=renderParas.map(parts=>parts.map(t=>{if(t.type!=="word")return t;const s=getSent(t.lemma,t.pos);
    return{...t,vader:s.vader??null,emotions:s.emolex?EMOTIONS.filter(e=>s.emolex[e]):[],vad:s.vad||null,frequency:freqMap[t.lemma]||0,relevance:relevanceMap[t.lemma]||0,community:commMap[t.lemma]??null,isTopN:ws.has(t.lemma)}}));
  return{enriched,freqPairs,freqMap,relevanceMap,maxFreq,maxRel,topWords,commMap,ng2,ng3}}

// ═══ CACHE ═══
const DB="texturas-cache",DBV=1,STO="assets";
function openDB(){return new Promise((r,j)=>{const q=indexedDB.open(DB,DBV);q.onupgradeneeded=()=>q.result.createObjectStore(STO);q.onsuccess=()=>r(q.result);q.onerror=()=>j(q.error)})}
async function cGet(k){try{const db=await openDB();return new Promise(r=>{const t=db.transaction(STO,"readonly");const q=t.objectStore(STO).get(k);q.onsuccess=()=>r(q.result||null);q.onerror=()=>r(null)})}catch{return null}}
async function cSet(k,v){try{const db=await openDB();return new Promise(r=>{const t=db.transaction(STO,"readwrite");t.objectStore(STO).put(v,k);t.oncomplete=()=>r(true);t.onerror=()=>r(false)})}catch{return false}}
async function loadAsset(key,path,bin,cb){const c=await cGet(key);if(c)return c;if(!ASSET_BASE_URL)return null;const b=ASSET_BASE_URL.endsWith("/")?ASSET_BASE_URL:ASSET_BASE_URL+"/";if(cb)cb("Fetching "+path+"...");try{const r=await fetch(b+path);if(!r.ok)return null;const d=bin?await r.arrayBuffer():await r.json();await cSet(key,d);return d}catch{return null}}

// ═══ WEAVE COMPONENTS ═══
function EmoBars({emotions,arousal,showEmo,showAro,enabledSlots}){
  if(!showEmo)return <span style={{position:"absolute",bottom:0,left:0,right:0,height:8,pointerEvents:"none"}}/>;
  const active=EMOTIONS.filter(e=>enabledSlots.has(e));
  const present=new Set(emotions);
  if(!active.length)return <span style={{position:"absolute",bottom:0,left:0,right:0,height:8,pointerEvents:"none"}}/>;
  const h=showAro?Math.max(2,Math.round((arousal??0.5)*8)):6;
  return <span style={{position:"absolute",bottom:0,left:0,right:0,height:8,display:"flex",gap:0,alignItems:"flex-end",pointerEvents:"none",padding:"0 1px"}}>
    {active.map(e=> <span key={e} style={{flex:1,height:present.has(e)?h:0,background:present.has(e)?EC[e]:"transparent",borderRadius:"1px 1px 0 0",opacity:0.85,minWidth:1,maxWidth:4}}/>)}</span>}

function WeaveTooltip({token,x,y}){
  if(!token||token.isStop)return null;const t=token;
  return <div style={{position:"fixed",left:Math.min(x+12,window.innerWidth-280),top:Math.min(y-10,window.innerHeight-300),zIndex:1000,padding:"10px 14px",background:"#1a1a1aee",border:"1px solid #444",borderRadius:6,fontFamily:"monospace",fontSize:11,color:"#ccc",pointerEvents:"none",maxWidth:260,lineHeight:1.7,backdropFilter:"blur(4px)"}}>
    <div style={{fontSize:13,color:"#4ecdc4",marginBottom:4,fontWeight:"bold"}}>{t.lemma} <span style={{color:"#666",fontWeight:"normal"}}>({t.pos})</span></div>
    {t.vader!==null&&<div>VADER: <span style={{color:t.vader>0.05?"#82e0aa":t.vader<-0.05?"#ff6b6b":"#888"}}>{t.vader>0?"+":""}{t.vader.toFixed(3)}</span></div>}
    {t.emotions.length>0&&<div>Emotions: {t.emotions.map(e=> <span key={e} style={{color:EC[e],marginRight:4}}>{e}</span>)}</div>}
    <div>Freq: <span style={{color:"#bb8fce"}}>{t.frequency}</span> · Relev: <span style={{color:"#4ecdc4"}}>{t.relevance.toFixed(1)}</span></div>
    {t.community!==null&&<div>Community: <span style={{color:CC[t.community%CC.length]}}>C{t.community+1}</span></div>}
    {t.vad&&<div>V={t.vad.v?.toFixed(2)} A={t.vad.a?.toFixed(2)} D={t.vad.d?.toFixed(2)}</div>}
  </div>}

function WeaveMinimap({enriched,layers,enabledSlots,showArousal,maxFreq,maxRel,scrollFrac,viewFrac,onSeek,height}){
  const cvRef=useRef();
  const flatWords=useMemo(()=>{const out=[];enriched.forEach((para,pi)=>{para.forEach(t=>{if(t.type==="word")out.push(t)});out.push(null)});return out},[enriched]);
  useEffect(()=>{const cv=cvRef.current;if(!cv)return;const ctx=cv.getContext("2d");const h=height||cv.parentElement?.clientHeight||400;
    cv.width=80;cv.height=h;ctx.fillStyle="#0d0d0d";ctx.fillRect(0,0,80,h);
    const total=flatWords.length;if(!total)return;
    const rowH=Math.max(1,Math.min(3,h/Math.ceil(total/16)));const cols=16,cw=Math.floor(80/cols);
    let row=0,col=0;
    flatWords.forEach(t=>{if(!t){row++;col=0;return}
      const y=row*rowH,x=col*cw;if(y>h)return;
      let c="#333";
      if(!t.isStop){
        if(layers.polarity&&t.vader!==null){c=t.vader>0.05?"#82e0aa":t.vader<-0.05?"#ff6b6b":"#666"}
        else if(layers.community&&t.community!==null){c=CC[t.community%CC.length]}
        else if(layers.emotion){const fe=t.emotions?.filter(e=>enabledSlots.has(e))||[];c=fe.length?EC[fe[0]]:"#555"}
        else if(layers.relevance){const rn=maxRel>0?Math.log(1+t.relevance)/Math.log(1+maxRel):0;const b=Math.round(50+rn*200);c="rgb("+b+","+Math.round(b*1.2)+","+b+")"}
        else if(layers.frequency){const fn=maxFreq>0?Math.log(1+t.frequency)/Math.log(1+maxFreq):0;const b=Math.round(40+fn*180);c="rgb("+b+","+b+","+b+")"}
        else c="#555"
      }else c="#222";
      ctx.fillStyle=c;ctx.fillRect(x+1,y,cw-1,Math.max(1,rowH-1));
      col++;if(col>=cols){col=0;row++}});
    const totalRows=row+1,mapH=totalRows*rowH;
    const vpY=scrollFrac*Math.min(mapH,h),vpH=Math.max(8,viewFrac*Math.min(mapH,h));
    ctx.fillStyle="rgba(78,205,196,0.15)";ctx.fillRect(0,vpY,80,vpH);
    ctx.strokeStyle="#4ecdc466";ctx.lineWidth=1;ctx.strokeRect(0.5,vpY+0.5,79,vpH-1);
  },[flatWords,layers,enabledSlots,showArousal,maxFreq,maxRel,scrollFrac,viewFrac,height]);
  const handleClick=useCallback(e=>{const cv=cvRef.current;if(!cv)return;const r=cv.getBoundingClientRect();const frac=(e.clientY-r.top)/r.height;onSeek(Math.max(0,Math.min(1,frac)))},[onSeek]);
  const dragRef=useRef(false);
  const onDown=useCallback(e=>{dragRef.current=true;handleClick(e)},[handleClick]);
  const onMove=useCallback(e=>{if(dragRef.current)handleClick(e)},[handleClick]);
  const onUp=useCallback(()=>{dragRef.current=false},[]);
  useEffect(()=>{window.addEventListener("mouseup",onUp);return ()=>window.removeEventListener("mouseup",onUp)},[onUp]);
  return <canvas ref={cvRef} style={{width:80,flexShrink:0,cursor:"pointer",borderLeft:"1px solid #2a2a2a",borderRight:"1px solid #2a2a2a"}} onMouseDown={onDown} onMouseMove={onMove}/>}

function WeaveReader({enriched,layers,highlightLemma,maxFreq,maxRel,onHover,onClick,enabledSlots,showArousal,scrollRef,onScroll,gridSize}){
  if(!enriched?.length) return <div style={{color:"#555",textAlign:"center",marginTop:60,fontSize:13}}>Run analysis to see annotated text.</div>;
  // Compute bin boundaries matching Vellum's tokenize (all words length>1)
  const gs=gridSize||10,cells=gs*gs;
  let totalWords=0;enriched.forEach(para=>para.forEach(t=>{if(t.type==="word"&&t.lower&&t.lower.length>1)totalWords++}));
  const base=Math.floor(totalWords/cells),extra=totalWords%cells;
  const binStarts=useMemo(()=>{const s=new Set();let pos=0;for(let i=0;i<cells;i++){s.add(pos);pos+=i<extra?base+1:base}return s},[totalWords,cells,base,extra]);
  const binLabel=(wi)=>{let pos=0;for(let i=0;i<cells;i++){const sz=i<extra?base+1:base;if(wi>=pos&&wi<pos+sz)return"["+(Math.floor(i/gs)+1)+","+(i%gs+1)+"]";pos+=sz}return null};

  let wordIdx=0;
  return <div ref={scrollRef} onScroll={onScroll} style={{flex:1,overflowY:"auto",padding:"24px 32px",lineHeight:2.6,wordSpacing:"0.06em"}}>
    {enriched.map((para,pi)=> <div key={pi} style={{marginBottom:20}}>{para.map((t,ti)=>{
      if(t.type!=="word") return <span key={ti} style={{fontFamily:"monospace"}}>{t.surface}</span>;
      const isWord=t.lower&&t.lower.length>1;
      const wi=isWord?wordIdx:-1;
      if(isWord)wordIdx++;
      const showMarker=isWord&&binStarts.has(wi);
      const marker=showMarker?binLabel(wi):null;

      if(t.isStop) return <span key={ti} style={{fontFamily:"monospace"}}>{showMarker&&<span data-bin={wi} title={marker} style={{display:"inline-block",width:0,height:"1.1em",borderLeft:"1.5px solid #4ecdc444",marginRight:2,verticalAlign:"middle",cursor:"pointer"}} onClick={e=>{e.target.scrollIntoView({behavior:"smooth",block:"start"})}}/>}<span style={{color:"#444"}}>{t.surface}</span></span>;
      const s={fontFamily:"monospace",cursor:"pointer",position:"relative",display:"inline-block",transition:"all 0.15s",padding:"1px 0px",paddingBottom:10,borderRadius:"2px"};
      s.color=layers.polarity&&t.vader!==null?(t.vader>0.05?"#82e0aa":t.vader<-0.05?"#ff6b6b":"#999"):"#ccc";
      if(layers.frequency){const fn=maxFreq>0?Math.log(1+t.frequency)/Math.log(1+maxFreq):0;s.opacity=0.25+fn*0.75}
      if(layers.relevance){const rn=maxRel>0?Math.log(1+t.relevance)/Math.log(1+maxRel):0;s.fontWeight=Math.round(100+rn*500)}else s.fontWeight=400;
      if(layers.community&&t.community!==null){s.backgroundColor=CC[t.community%CC.length]+"1a"}
      if(highlightLemma&&t.lemma===highlightLemma){s.outline="2px solid #4ecdc4";s.outlineOffset="3px";s.borderRadius="3px";s.textShadow="0 0 8px #4ecdc466";if(!s.backgroundColor)s.backgroundColor="#4ecdc40d"}
      const filtEmo=layers.emotion?t.emotions.filter(e=>enabledSlots.has(e)):[];
      return <span key={ti} style={{fontFamily:"monospace"}}>{showMarker&&<span data-bin={wi} title={marker} style={{display:"inline-block",width:0,height:"1.1em",borderLeft:"1.5px solid #4ecdc466",marginRight:2,verticalAlign:"middle",cursor:"pointer"}} onClick={e=>{e.target.scrollIntoView({behavior:"smooth",block:"start"})}}/>}<span style={s} onMouseEnter={e=>onHover(t,e.clientX,e.clientY)} onMouseMove={e=>onHover(t,e.clientX,e.clientY)} onMouseLeave={()=>onHover(null,0,0)} onClick={()=>onClick(t.lemma)}>
        {t.surface}<EmoBars emotions={filtEmo} arousal={t.vad?.a??null} showEmo={layers.emotion} showAro={showArousal} enabledSlots={enabledSlots}/></span></span>})}</div>)}</div>}

function WeaveWordPanel({result,topN,highlightLemma,onClickWord,ngMode,setNgMode}){
  const{freqPairs,relevanceMap,maxFreq,maxRel,ng2,ng3}=result;
  const[sortBy,setSortBy]=useState("freq");
  const isUni=ngMode==="1";const items=isUni?freqPairs.slice(0,topN):ngMode==="2"?ng2.slice(0,topN):ng3.slice(0,topN);
  const sorted=useMemo(()=>{if(!isUni||sortBy==="freq")return items;return[...items].sort((a,b)=>(relevanceMap[b[0]]||0)-(relevanceMap[a[0]]||0))},[items,sortBy,isUni,relevanceMap]);
  const maxV=items[0]?.[1]||1;
  const gMR=isUni?Math.max(...items.map(([w])=>relevanceMap[w]||0),1):1;
  return <div style={{display:"flex",flexDirection:"column",gap:1,minWidth:0,height:"100%"}}>
    <div style={{display:"flex",gap:0,marginBottom:2,border:"1px solid #333",borderRadius:3,overflow:"hidden"}}>{[["1","1"],["2","2"],["3","3"]].map(([v,l])=> <button key={v} onClick={()=>{setNgMode(v);setSortBy("freq")}} style={{flex:1,padding:"4px 0",background:ngMode===v?"#bb8fce":"#1a1a1a",color:ngMode===v?"#111":"#666",border:"none",cursor:"pointer",fontSize:10,fontFamily:"monospace",fontWeight:ngMode===v?"bold":"normal"}}>{l}</button>)}</div>
    <button onClick={()=>onClickWord(null)} style={{padding:"4px 8px",marginBottom:2,background:!highlightLemma?"#4ecdc4":"#1a1a1a",color:!highlightLemma?"#111":"#999",border:"1px solid "+(!highlightLemma?"#4ecdc4":"#333"),borderRadius:4,cursor:"pointer",fontSize:11,fontFamily:"monospace",fontWeight:!highlightLemma?"bold":"normal",textAlign:"left"}}>All</button>
    {isUni&&<div style={{display:"flex",gap:0,marginBottom:3,border:"1px solid #333",borderRadius:3,overflow:"hidden"}}>{[["freq","Freq"],["relevance","Relev"]].map(([v,l])=> <button key={v} onClick={()=>setSortBy(v)} style={{flex:1,padding:"3px 0",background:sortBy===v?"#45b7d1":"#1a1a1a",color:sortBy===v?"#111":"#666",border:"none",cursor:"pointer",fontSize:9,fontFamily:"monospace"}}>{l}</button>)}</div>}
    <div style={{flex:1,overflowY:"auto",display:"flex",flexDirection:"column",gap:1}}>{sorted.map(([w,c])=>{const rel=isUni?(relevanceMap[w]||0):0,isHL=highlightLemma===w,pr=c>0;
      return <button key={w} onClick={()=>onClickWord(w)} style={{padding:"3px 7px",background:isHL?"#1a2a2a":"#151515",color:isHL?"#ddd":pr?"#777":"#444",border:"1px solid "+(isHL?"#4ecdc466":"#222"),borderRadius:3,cursor:"pointer",fontSize:10,fontFamily:"monospace",textAlign:"left",display:"flex",flexDirection:"column",gap:1,opacity:pr?1:.35}}>
        <div style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",width:"100%"}}>{w}</div>
        <div style={{display:"flex",alignItems:"center",gap:4,width:"100%",height:12}}><div style={{flex:1,height:3,background:"#222",borderRadius:1,overflow:"hidden"}}><div style={{width:(c/maxV*100)+"%",height:"100%",background:isHL?"#4ecdc4":"#444",borderRadius:1,opacity:isHL?.8:.3}}/></div><span style={{width:24,textAlign:"right",fontSize:9,color:isHL?"#aaa":"#555",flexShrink:0}}>{c||""}</span></div>
        {isUni&&<div style={{display:"flex",alignItems:"center",gap:4,width:"100%",height:12}}><div style={{flex:1,height:3,background:"#222",borderRadius:1,overflow:"hidden"}}><div style={{width:(rel/gMR*100)+"%",height:"100%",background:isHL?"#45b7d1":"#444",borderRadius:1,opacity:isHL?.8:.3}}/></div><span style={{width:24,textAlign:"right",fontSize:9,color:isHL?"#7ab8cc":"#444",flexShrink:0}}>{rel?rel.toFixed(0):""}</span></div>}
      </button>})}</div>
    <div style={{display:"flex",gap:8,marginTop:3,fontSize:9,color:"#888"}}><span><span style={{color:"#4ecdc4"}}>—</span> freq</span>{isUni&&<span><span style={{color:"#45b7d1"}}>—</span> relev</span>}</div>
  </div>}

function EmoToggle({enabledSlots,setEnabledSlots}){
  return (<div style={{display:"inline-grid",gridTemplateColumns:"repeat(3,10px)",gap:1}}>
    {EMO_LAYOUT.map(({r,c,emo},i)=>{const key=emo||"center",on=enabledSlots.has(key),col=emo?EC[emo]:"#ffffff";
      return (<div key={i} onClick={()=>setEnabledSlots(prev=>{const n=new Set(prev);if(n.has(key))n.delete(key);else n.add(key);return n})} title={emo||"relevance"} style={{width:10,height:10,borderRadius:1,background:on?col:"#555",opacity:on?1:.5,cursor:"pointer",border:"1px solid "+(on?col+"66":"#444")}}/>)})}</div>)}

// ═══ MAIN ═══
function WeaveStandalone(){
  const[docs,setDocs]=useState([{id:"d1",label:"Document 1",text:""}]);
  const[activeInputDoc,setActiveInputDoc]=useState("d1");
  const[tab,setTab]=useState("input");
  const[topN,setTopN]=useState(25),[wnDepth,setWnDepth]=useState(2),[decay,setDecay]=useState(0.5),[flow,setFlow]=useState("bi"),[winSize]=useState(5);
  const[weavePerDoc,setWeavePerDoc]=useState({});
  const[loading,setLoading]=useState(false),[msg,setMsg]=useState("");
  const[wLayers,setWLayers]=useState({polarity:true,emotion:true,frequency:false,relevance:false,community:false});
  const[showArousal,setShowArousal]=useState(false);
  const[enabledSlots,setEnabledSlots]=useState(new Set([...EMOTIONS,"center"]));
  const[gridSize,setGridSize]=useState(10);
  const[wHighlight,setWHighlight]=useState(null);
  const[wHovTok,setWHovTok]=useState(null);
  const[wHovPos,setWHovPos]=useState({x:0,y:0});
  const[wNgMode,setWNgMode]=useState("1");
  const[wActiveDoc,setWActiveDoc]=useState(null);
  const[showParams,setShowParams]=useState(false);
  const readerRef=useRef();
  const[scrollFrac,setScrollFrac]=useState(0);
  const[viewFrac,setViewFrac]=useState(1);
  const[contentH,setContentH]=useState(400);
  const handleReaderScroll=useCallback(()=>{const el=readerRef.current;if(!el)return;const sh=el.scrollHeight,ch=el.clientHeight;setScrollFrac(sh>ch?el.scrollTop/(sh-ch):0);setViewFrac(sh>0?ch/sh:1)},[]);
  const handleMinimapSeek=useCallback(frac=>{const el=readerRef.current;if(!el)return;const sh=el.scrollHeight,ch=el.clientHeight;el.scrollTop=frac*(sh-ch)},[]);
  const contentRef=useRef();
  useEffect(()=>{if(!contentRef.current)return;const ro=new ResizeObserver(en=>{setContentH(en[0].contentRect.height)});ro.observe(contentRef.current);return ()=>ro.disconnect()},[]);

  const eng=useRef({pos:new POSTagger(),lem:new Lemmatizer(),syn:new SynsetEngine(),sent:new SentimentScorer()});
  useEffect(()=>{if(!ASSET_BASE_URL)return;let c=false;(async()=>{const e=eng.current;
    const pd=await loadAsset("w-p","wordnet/pos-lookup.json",false,setMsg);if(pd&&!c)e.pos.load(pd);
    const ld=await loadAsset("w-l","wordnet/lemmatizer.json",false,setMsg);if(ld&&!c)e.lem.load(ld);
    const sd=await loadAsset("w-s","wordnet/synsets.json",false,setMsg);if(sd&&!c)e.syn.load(sd);
    const el=await loadAsset("l-e","lexicons/nrc-emolex.json",false,setMsg);if(el&&!c)e.sent.lEl(el);
    const ni=await loadAsset("l-i","lexicons/nrc-intensity.json",false,setMsg);if(ni&&!c)e.sent.lInt(ni);
    const nv=await loadAsset("l-v","lexicons/nrc-vad.json",false,setMsg);if(nv&&!c)e.sent.lVad(nv);
    const va=await loadAsset("l-d","lexicons/vader.json",false,setMsg);if(va&&!c)e.sent.lVdr(va);
    const sw=await loadAsset("l-s","lexicons/sentiwordnet.json",false,setMsg);if(sw&&!c)e.sent.lSwn(sw);
    if(!c)setMsg("")})();return ()=>{c=true}},[]);

  const addDoc=()=>{const id="d"+Date.now();setDocs(d=>[...d,{id,label:"Document "+(d.length+1),text:""}]);setActiveInputDoc(id)};
  const rmDoc=id=>{if(docs.length<=1)return;setDocs(d=>d.filter(x=>x.id!==id));if(activeInputDoc===id)setActiveInputDoc(docs[0]?.id)};
  const updDoc=(id,f,v)=>setDocs(d=>d.map(x=>x.id===id?{...x,[f]:v}:x));
  const handleFiles=files=>{Array.from(files).forEach(f=>{if(!f.name.endsWith(".txt"))return;const id="d"+Date.now()+"_"+Math.random().toString(36).slice(2,6);const reader=new FileReader();reader.onload=ev=>{setDocs(d=>[...d.filter(x=>x.text.trim()),{id,label:f.name.replace(".txt",""),text:ev.target.result}])};reader.readAsText(f)})};
  const parseSep=text=>{const parts=text.split(/---DOC(?::?\s*([^-]*))?\s*---/i),result=[];let pl=null;for(let i=0;i<parts.length;i++){const t=parts[i]?.trim();if(!t)continue;if(i%2===1)pl=t;else{result.push({id:"d"+Date.now()+"_"+i,label:pl||"Document "+(result.length+1),text:t});pl=null}}return result.length>1?result:null};

  const validDocs=docs.filter(d=>d.text.trim()),analyzedIds=Object.keys(weavePerDoc);

  const runAnalysis=useCallback(()=>{if(!validDocs.length)return;setLoading(true);setMsg("Analyzing...");setTimeout(()=>{const e=eng.current;
    const wdr={};validDocs.forEach(d=>{wdr[d.id]=analyzeForWeave(d.text,e,topN,winSize,wnDepth,decay,flow)});
    setWeavePerDoc(wdr);setWActiveDoc(validDocs[0].id);setWHighlight(null);
    setLoading(false);setMsg("");setTab("weave");setTimeout(handleReaderScroll,100)},50)},[docs,topN,wnDepth,decay,flow,winSize]);

  const rerunFlow=v=>{setFlow(v);if(!validDocs.length||!analyzedIds.length)return;setTimeout(()=>{const e=eng.current;const wdr={};validDocs.forEach(d=>{wdr[d.id]=analyzeForWeave(d.text,e,topN,winSize,wnDepth,decay,v)});setWeavePerDoc(wdr)},50)};
  const rerunTopN=useCallback(n=>{setTopN(n);if(!validDocs.length||!analyzedIds.length)return;setTimeout(()=>{const e=eng.current;const wdr={};validDocs.forEach(d=>{wdr[d.id]=analyzeForWeave(d.text,e,n,winSize,wnDepth,decay,flow)});setWeavePerDoc(wdr)},50)},[docs,wnDepth,decay,flow,winSize]);
  const rerunDecay=useCallback(d=>{setDecay(d);if(!validDocs.length||!analyzedIds.length)return;setTimeout(()=>{const e=eng.current;const wdr={};validDocs.forEach(doc=>{wdr[doc.id]=analyzeForWeave(doc.text,e,topN,winSize,wnDepth,d,flow)});setWeavePerDoc(wdr)},50)},[docs,topN,wnDepth,flow,winSize]);
  const toggleWLayer=id=>{setWLayers(prev=>{const next={...prev,[id]:!prev[id]};if(Object.values(next).every(v=>!v))return prev;return next})};

  const activeWR=wActiveDoc?weavePerDoc[wActiveDoc]:null;
  const hasMarkers=docs.find(d=>d.id===activeInputDoc)?.text?.includes("---DOC")||false;

  return (<div style={{background:"#111",color:"#ddd",minHeight:"100vh",fontFamily:"monospace",display:"flex",flexDirection:"column"}}>
    {/* Header */}
    <div style={{padding:"12px 20px",borderBottom:"1px solid #2a2a2a",display:"flex",alignItems:"center",gap:12}}>
      <span style={{fontSize:18,color:"#4ecdc4",fontWeight:"bold"}}>⬡ Texturas</span><span style={{fontSize:11,color:"#555"}}>Weave standalone</span>
      <div style={{marginLeft:"auto",display:"flex",gap:8}}>
        {eng.current.sent.ready&&<span style={{fontSize:10,color:"#f7dc6f"}}>● sent</span>}
        {eng.current.pos.ready&&<span style={{fontSize:10,color:"#bb8fce"}}>● nlp</span>}
        {eng.current.syn.ready&&<span style={{fontSize:10,color:"#45b7d1"}}>● wn</span>}
      </div>
    </div>

    {/* Tab bar */}
    <div style={{display:"flex",borderBottom:"1px solid #2a2a2a"}}>
      {[{id:"input",l:"Input"},{id:"weave",l:"Weave"}].map(t=> <button key={t.id} onClick={()=>setTab(t.id)} style={{padding:"10px 14px",background:tab===t.id?"#1a1a1a":"transparent",color:tab===t.id?"#4ecdc4":"#888",border:"none",borderBottom:tab===t.id?"2px solid #4ecdc4":"2px solid transparent",cursor:"pointer",fontSize:12,fontFamily:"monospace"}}>{t.l}</button>)}
    </div>

    {/* Doc selector */}
    {tab==="weave"&&analyzedIds.length>0&&(<div style={{display:"flex",gap:4,padding:"8px 20px",borderBottom:"1px solid #2a2a2a",background:"#151515"}}>
      {validDocs.filter(d=>weavePerDoc[d.id]).map(d=> <button key={d.id} onClick={()=>{setWActiveDoc(d.id);setWHighlight(null)}} style={{padding:"4px 12px",borderRadius:3,border:"1px solid "+(wActiveDoc===d.id?"#45b7d1":"#333"),background:wActiveDoc===d.id?"#45b7d1":"#1a1a1a",color:wActiveDoc===d.id?"#111":"#888",fontSize:11,fontFamily:"monospace",cursor:"pointer"}}>{d.label} <span style={{opacity:.6,fontSize:9}}>({d.text.trim().split(/\s+/).length.toLocaleString()})</span></button>)}</div>)}




    {/* Content */}
    <div style={{flex:1,padding:tab==="weave"?0:"16px 20px",overflowY:"auto"}}>

      {/* INPUT TAB */}
      {tab==="input"&&(<div style={{maxWidth:800,margin:"0 auto"}}>
        <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}>
          <span style={{fontSize:13,color:"#aaa"}}>Documents ({docs.length})</span>
          <button onClick={addDoc} style={{padding:"4px 12px",background:"#2a2a2a",color:"#4ecdc4",border:"1px solid #444",borderRadius:4,fontSize:12,fontFamily:"monospace",cursor:"pointer"}}>+ Add</button>
          <label style={{padding:"4px 12px",background:"#2a2a2a",color:"#45b7d1",border:"1px solid #444",borderRadius:4,fontSize:12,fontFamily:"monospace",cursor:"pointer"}}>Upload .txt<input type="file" multiple accept=".txt" style={{display:"none"}} onChange={ev=>handleFiles(ev.target.files)}/></label>
        </div>
        <div style={{display:"flex",gap:4,marginBottom:12,flexWrap:"wrap"}}>{docs.map(d=> <button key={d.id} onClick={()=>setActiveInputDoc(d.id)} style={{padding:"5px 12px",borderRadius:4,border:"1px solid "+(activeInputDoc===d.id?"#45b7d1":"#333"),background:activeInputDoc===d.id?"#1a2a2a":"#1a1a1a",color:activeInputDoc===d.id?"#45b7d1":"#888",fontSize:12,fontFamily:"monospace",cursor:"pointer"}}>{d.label}{docs.length>1&&<span onClick={ev=>{ev.stopPropagation();rmDoc(d.id)}} style={{marginLeft:8,color:"#666",cursor:"pointer"}}>×</span>}</button>)}</div>
        {docs.filter(d=>d.id===activeInputDoc).map(d=> <div key={d.id}>
          <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:8}}>
            <input type="text" value={d.label} onChange={ev=>updDoc(d.id,"label",ev.target.value)} style={{width:300,padding:"6px 10px",background:"#1a1a1a",border:"1px solid #444",borderRadius:4,color:"#ccc",fontSize:13,fontFamily:"monospace",boxSizing:"border-box"}}/>
            {hasMarkers&&<button onClick={()=>{const p=parseSep(d.text);if(p){setDocs(p);setActiveInputDoc(p[0].id)}}} style={{padding:"6px 14px",background:"#2a2a1a",color:"#f0b27a",border:"1px solid #f0b27a44",borderRadius:4,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>Split</button>}
          </div>
          <textarea value={d.text} onChange={ev=>updDoc(d.id,"text",ev.target.value)} onPaste={ev=>{const t=ev.clipboardData.getData("text");if(t.includes("---DOC")){ev.preventDefault();const p=parseSep(t);if(p){setDocs(p);setActiveInputDoc(p[0].id)}}}} placeholder={"Paste text here...\n\nUse ---DOC: Label--- separators for multiple documents"} style={{width:"100%",height:"calc(100vh - 400px)",background:"#0d0d0d",border:"1px solid #2a2a2a",borderRadius:6,color:"#ccc",padding:16,fontSize:13,fontFamily:"monospace",resize:"none",boxSizing:"border-box",lineHeight:1.6}}/>
        </div>)}
        <div style={{marginTop:12,display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}}>
          <button onClick={()=>setShowParams(!showParams)} style={{padding:"6px 12px",background:"#1a1a1a",color:"#888",border:"1px solid #333",borderRadius:4,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>Parameters {showParams?"▾":"▸"}</button>
          <button onClick={runAnalysis} disabled={!validDocs.length||loading} style={{padding:"8px 20px",background:validDocs.length&&!loading?"#4ecdc4":"#333",color:validDocs.length&&!loading?"#111":"#666",border:"none",borderRadius:4,cursor:validDocs.length&&!loading?"pointer":"default",fontFamily:"monospace",fontSize:13,fontWeight:"bold"}}>Analyze {validDocs.length} doc{validDocs.length!==1?"s":""} →</button>
          {msg&&<span style={{fontSize:10,color:"#f7dc6f"}}>{msg}</span>}
        </div>
        {showParams&&(<div style={{marginTop:10,padding:14,background:"#1a1a1a",borderRadius:6,border:"1px solid #333",display:"flex",gap:20,flexWrap:"wrap",alignItems:"flex-end"}}>
          <div><label style={{fontSize:10,color:"#888",display:"block",marginBottom:3}}>Top N: {topN}</label><input type="range" min={10} max={50} value={topN} onChange={ev=>setTopN(+ev.target.value)} style={{width:100}}/></div>
          <div><label style={{fontSize:10,color:"#888",display:"block",marginBottom:3}}>WN depth: {wnDepth}</label><input type="range" min={1} max={3} value={wnDepth} onChange={ev=>setWnDepth(+ev.target.value)} style={{width:80}}/></div>
          <div><label style={{fontSize:10,color:"#888",display:"block",marginBottom:3}}>Decay: {decay.toFixed(2)}</label><input type="range" min={30} max={80} value={decay*100} onChange={ev=>rerunDecay(+ev.target.value/100)} style={{width:100}}/></div>
          <div><label style={{fontSize:10,color:"#888",display:"block",marginBottom:3}}>Flow:</label><div style={{display:"flex",gap:0,border:"1px solid #333",borderRadius:3,overflow:"hidden"}}>{[["bi","Bi"],["up","Up ↑"],["down","Dn ↓"]].map(([v,l])=> <button key={v} onClick={()=>setFlow(v)} style={{padding:"4px 8px",background:flow===v?"#45b7d1":"#1a1a1a",color:flow===v?"#111":"#666",border:"none",cursor:"pointer",fontSize:10,fontFamily:"monospace"}}>{l}</button>)}</div></div>
        </div>)}
      </div>)}

      {/* WEAVE TAB — content area matches Vellum: flex + gap:10 + alignItems:stretch */}
      {tab==="weave"&&activeWR&&<div style={{maxWidth:1100,margin:"0 auto",padding:"16px 20px"}}>
        <div style={{display:"flex",gap:8,marginBottom:12,alignItems:"center",height:36}}>
          {LAYER_CFG.map(l=> <button key={l.id} onClick={()=>toggleWLayer(l.id)} title={l.desc} style={{padding:"5px 10px",borderRadius:4,fontSize:11,fontFamily:"monospace",cursor:"pointer",background:wLayers[l.id]?l.color+"22":"#1a1a1a",color:wLayers[l.id]?l.color:"#555",border:"1px solid "+(wLayers[l.id]?l.color:"#333"),transition:"all 0.15s"}}>{l.label}</button>)}
          <div style={{width:40,flexShrink:0,display:"flex",justifyContent:"flex-start"}}>{wLayers.emotion&&<EmoToggle enabledSlots={enabledSlots} setEnabledSlots={setEnabledSlots}/>}</div>
          <div style={{width:1,height:22,background:"#333"}}/>
          <div style={{display:"flex",gap:0,border:"1px solid #333",borderRadius:4,overflow:"hidden"}}>{[10,20,30].map(g=> <button key={g} onClick={()=>setGridSize(g)} style={{padding:"5px 11px",background:gridSize===g?"#bb8fce":"#1a1a1a",color:gridSize===g?"#111":"#666",border:"none",cursor:"pointer",fontSize:11,fontFamily:"monospace",fontWeight:gridSize===g?"bold":"normal"}}>{g}²</button>)}</div>
          <div style={{marginLeft:"auto",display:"flex",gap:8,alignItems:"center"}}>
            <button onClick={()=>setShowArousal(!showArousal)} style={{padding:"5px 10px",background:showArousal?"#2a2a1a":"#1a1a1a",color:showArousal?"#f7dc6f":"#555",border:"1px solid "+(showArousal?"#f7dc6f44":"#333"),borderRadius:4,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>Arousal</button>
            <div style={{width:1,height:22,background:"#333"}}/>
            <div style={{display:"flex",gap:0,border:"1px solid #333",borderRadius:4,overflow:"hidden"}}>{[["bi","Bi"],["up","↑"],["down","↓"]].map(([v,l])=> <button key={v} onClick={()=>rerunFlow(v)} style={{padding:"5px 9px",background:flow===v?"#45b7d1":"#1a1a1a",color:flow===v?"#111":"#666",border:"none",cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>{l}</button>)}</div>
            <div style={{width:1,height:22,background:"#333"}}/>
            <div style={{display:"flex",alignItems:"center",gap:5}}><span style={{fontSize:10,color:"#666"}}>N:</span><input type="range" min={10} max={50} value={topN} onChange={ev=>rerunTopN(+ev.target.value)} style={{width:60}}/><span style={{fontSize:10,color:"#aaa",width:16}}>{topN}</span></div>
            <div style={{width:1,height:22,background:"#333"}}/>
            <div style={{display:"flex",alignItems:"center",gap:5}}><span style={{fontSize:10,color:"#666"}}>decay:</span><input type="range" min={30} max={80} value={decay*100} onChange={ev=>rerunDecay(+ev.target.value/100)} style={{width:50}}/><span style={{fontSize:10,color:"#aaa",width:24}}>{decay.toFixed(2)}</span></div>
          </div>
        </div>
        <div ref={contentRef} style={{display:"flex",gap:10,alignItems:"stretch",height:540,overflow:"hidden"}}>
          <WeaveReader enriched={activeWR.enriched} layers={wLayers} highlightLemma={wHighlight} maxFreq={activeWR.maxFreq} maxRel={activeWR.maxRel} onHover={(t,x,y)=>{setWHovTok(t);setWHovPos({x,y})}} onClick={lem=>setWHighlight(prev=>prev===lem?null:lem)} enabledSlots={enabledSlots} showArousal={showArousal} scrollRef={readerRef} onScroll={handleReaderScroll} gridSize={gridSize}/>
          <WeaveMinimap enriched={activeWR.enriched} layers={wLayers} enabledSlots={enabledSlots} showArousal={showArousal} maxFreq={activeWR.maxFreq} maxRel={activeWR.maxRel} scrollFrac={scrollFrac} viewFrac={viewFrac} onSeek={handleMinimapSeek} height={contentH}/>
          <div style={{width:160,flexShrink:0,display:"flex",flexDirection:"column"}}>
            <div style={{fontSize:10,color:"#888",marginBottom:3}}>Top {topN} · click to highlight</div>
            <WeaveWordPanel result={activeWR} topN={topN} highlightLemma={wHighlight} onClickWord={w=>setWHighlight(prev=>prev===w?null:w)} ngMode={wNgMode} setNgMode={setWNgMode}/>
          </div>
        </div>
        <div style={{display:"flex",gap:14,marginTop:10,padding:"8px 12px",background:"#0d0d0d",borderRadius:4,border:"1px solid #1a1a1a",fontSize:11,color:"#555",flexWrap:"wrap"}}>
          {wLayers.polarity&&<span><span style={{color:"#82e0aa"}}>■</span>/<span style={{color:"#ff6b6b"}}>■</span> polarity</span>}
          {wLayers.emotion&&<span><span style={{color:"#f0b27a"}}>—</span> emotion</span>}
          {showArousal&&<span><span style={{color:"#f7dc6f"}}>━</span> arousal</span>}
          {wLayers.frequency&&<span><span style={{color:"#bb8fce"}}>○</span> brightness</span>}
          {wLayers.relevance&&<span><span style={{color:"#4ecdc4",fontWeight:600}}>B</span> weight</span>}
          {wLayers.community&&<span style={{background:"#4ecdc41a",padding:"0 4px",borderRadius:2}}>community</span>}
        </div>
      </div>}
      {tab==="weave"&&!activeWR&&<div style={{flex:1,display:"flex",alignItems:"center",justifyContent:"center"}}><div style={{textAlign:"center",color:"#555"}}><div style={{fontSize:14,marginBottom:8}}>← Analyze documents first.</div><div style={{fontSize:11,color:"#444"}}>Six analytical layers projected onto the text itself.</div></div></div>}
    </div>

    <WeaveTooltip token={wHovTok} x={wHovPos.x} y={wHovPos.y}/>
    {loading&&<div style={{position:"fixed",bottom:20,left:"50%",transform:"translateX(-50%)",padding:"8px 20px",background:"#1a1a1aee",border:"1px solid #444",borderRadius:6,fontSize:11,color:"#f7dc6f",fontFamily:"monospace",zIndex:999}}>{msg||"Processing..."}</div>}
  </div>)}
