const { useState, useRef, useEffect, useCallback, useMemo } = React;

var ASSET_BASE_URL = "https://epsidebox.github.io/texturas/assets/";
var STOP_WORDS = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","shall","should","may","might","can","could","this","that","these","those","it","its","i","me","my","mine","we","us","our","ours","you","your","yours","he","him","his","she","her","hers","they","them","their","theirs","what","which","who","whom","whose","where","when","how","why","if","then","else","so","because","as","than","too","very","just","about","above","after","again","all","also","am","any","before","below","between","both","during","each","few","further","get","got","here","into","more","most","much","only","other","out","over","own","same","some","such","there","through","under","until","up","while","down","like","make","many","now","off","once","one","onto","per","since","still","take","thing","things","think","two","upon","well","went","yeah","yes","oh","um","uh","ah","okay","ok","really","actually","basically","literally","right","know","going","go","come","say","said","tell","told","see","look","want","need","way","back","even","around","something","anything","everything","someone","anyone","everyone","always","sometimes","often","already","yet","kind","sort","lot","bit","little","big","good","great","long","new","old","first","last","next","time","times","day","days","year","years","people","person","part","place","point","work","life","world","hand","week","end","case","fact","area","number","given","able","made","set","used","using","another","different","example","enough","however","important","include","including","keep","large","whether","without","within","along","become","across","among","toward","towards","though","although","either","rather","several","certain","less","likely","began","begun","brought","thus","therefore","hence","moreover","furthermore","nevertheless","nonetheless","meanwhile","accordingly","consequently","subsequently"]);
var NEGATION = new Set(["not","no","never","neither","nor","don't","doesn't","didn't","won't","wouldn't","can't","couldn't","shouldn't","isn't","aren't","wasn't","weren't","haven't","hasn't","hadn't","cannot","nothing","nobody","none","nowhere"]);
var EMOTIONS = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"];
var EC = {anger:"#ff6b6b",anticipation:"#f7dc6f",disgust:"#bb8fce",fear:"#85c1e9",joy:"#82e0aa",sadness:"#45b7d1",surprise:"#f0b27a",trust:"#4ecdc4"};
var CC = ["#ff6b6b","#4ecdc4","#45b7d1","#f7dc6f","#bb8fce","#82e0aa","#f0b27a","#85c1e9","#f1948a","#73c6b6"];
var LAYER_CFG = [{id:"polarity",label:"Polarity",color:"#82e0aa"},{id:"emotion",label:"Emotion",color:"#f0b27a"},{id:"frequency",label:"Frequency",color:"#bb8fce"},{id:"relevance",label:"Relevance",color:"#4ecdc4"},{id:"community",label:"Community",color:"#45b7d1"}];
var EMO_LAYOUT = [{r:0,c:0,emo:"anticipation"},{r:0,c:1,emo:"joy"},{r:0,c:2,emo:"trust"},{r:1,c:0,emo:"anger"},{r:1,c:1,emo:null},{r:1,c:2,emo:"fear"},{r:2,c:0,emo:"disgust"},{r:2,c:1,emo:"sadness"},{r:2,c:2,emo:"surprise"}];
var SFX=[["tion","n"],["sion","n"],["ment","n"],["ness","n"],["ity","n"],["ance","n"],["ence","n"],["ism","n"],["ist","n"],["ting","v"],["sing","v"],["ning","v"],["ing","v"],["ated","v"],["ized","v"],["ed","v"],["ize","v"],["ify","v"],["ously","r"],["ively","r"],["fully","r"],["ally","r"],["ly","r"],["ful","a"],["ous","a"],["ive","a"],["able","a"],["ible","a"],["ical","a"],["less","a"],["al","a"]];

function mkPOS(){var o={lk:null,ready:false};o.load=function(d){o.lk=d;o.ready=true};o.tag=function(w){var l=w.toLowerCase();if(o.lk&&o.lk[l])return o.lk[l];for(var i=0;i<SFX.length;i++)if(l.endsWith(SFX[i][0]))return SFX[i][1];return"n"};return o}
function mkLem(){var o={exc:null,rl:null,ready:false};o.load=function(d){o.exc=d.exceptions;o.rl=d.rules;o.ready=true};o.lemmatize=function(w,pos){var l=w.toLowerCase();if(o.exc&&o.exc[pos]&&o.exc[pos][l])return o.exc[pos][l];var rules=(o.rl&&o.rl[pos])||[];for(var i=0;i<rules.length;i++)if(l.endsWith(rules[i][0])&&l.length>rules[i][0].length)return l.slice(0,-rules[i][0].length)+rules[i][1];return l};return o}
function mkSyn(){var o={d:null,ready:false};o.load=function(d){o.d=d;o.ready=true};o.getRels=function(w,p){var e=o.d?o.d[w+"#"+p]:null;if(!e)return{h:[],y:[],m:[]};return{h:e.hypernyms||[],y:e.hyponyms||[],m:e.meronyms||[]}};o.directed=function(w,p,flow){var r=o.getRels(w,p);if(flow==="up")return Array.from(new Set([].concat(r.h,r.m)));if(flow==="down")return Array.from(new Set([].concat(r.y,r.m)));return Array.from(new Set([].concat(r.h,r.y,r.m)))};return o}
function mkSent(){var o={el:null,int:null,vad:null,vdr:null,swn:null,ready:false};function ck(){o.ready=!!(o.el||o.int||o.vad||o.vdr||o.swn)}o.lEl=function(d){o.el=d;ck()};o.lInt=function(d){o.int=d;ck()};o.lVad=function(d){o.vad=d;ck()};o.lVdr=function(d){o.vdr=d;ck()};o.lSwn=function(d){o.swn=d;ck()};o.score=function(lem,pos){var r={lemma:lem,pos:pos};if(o.el&&o.el[lem])r.emolex=o.el[lem];if(o.int&&o.int[lem])r.intensity=o.int[lem];if(o.vad&&o.vad[lem])r.vad=o.vad[lem];if(o.vdr&&o.vdr[lem]!==undefined)r.vader=o.vdr[lem];var k=lem+"#"+pos;if(o.swn&&o.swn[k])r.swn=o.swn[k];else{var tries=["n","v","a","r"];for(var i=0;i<tries.length;i++){var v=o.swn?o.swn[lem+"#"+tries[i]]:null;if(v){r.swn=v;break}}}r.has=!!(r.emolex||r.intensity||r.vad||r.vader!==undefined||r.swn);return r};return o}

function spreadAct(fMap,syn,pos,depth,decay,flow){if(!syn.ready)return Object.assign({},fMap);var corpus=new Set(Object.keys(fMap));var scores={};corpus.forEach(function(l){scores[l]=fMap[l]||0});Object.entries(fMap).forEach(function(pair){var l=pair[0],f=pair[1];if(f===0)return;var p=pos.ready?pos.tag(l):"n";var front=[{w:l,p:p}];var vis=new Set([l]);for(var h=1;h<=depth;h++){var nf=[];front.forEach(function(nd){var tgts=syn.directed(nd.w,nd.p,flow);tgts.forEach(function(t){if(vis.has(t))return;vis.add(t);var amt=f*Math.pow(decay,h);if(corpus.has(t))scores[t]=(scores[t]||0)+amt;nf.push({w:t,p:pos.ready?pos.tag(t):"n"})})});front=nf}});return scores}

function getNg(ts,n){var g=[];for(var i=0;i<=ts.length-n;i++)g.push(ts.slice(i,i+n).join(" "));return _.chain(g).countBy().toPairs().sortBy(1).reverse().value()}

function analyzeForWeave(text,eng,topN,winSize,wnDepth,decay,flow){
  var norm=text.replace(/[\u2018\u2019\u201A\u201B]/g,"'").replace(/[\u201C\u201D\u201E\u201F]/g,'"');
  var paras=norm.split(/\n\s*\n+/).filter(function(p){return p.trim()});var allWords=[];
  var renderParas=paras.map(function(para){var parts=[];var rx=/([\w'-]+)|(\s+)|([^\w\s'-]+)/g;var m;while((m=rx.exec(para))!==null){if(m[1]){var surface=m[1],lower=surface.toLowerCase();var isStop=STOP_WORDS.has(lower)&&!NEGATION.has(lower);var p=eng.pos.ready?eng.pos.tag(lower):"n";var lemma=eng.lem.ready?eng.lem.lemmatize(lower,p):lower;if(!isStop)allWords.push(lemma);parts.push({type:"word",surface:surface,lower:lower,lemma:lemma,pos:p,isStop:isStop})}else parts.push({type:"other",surface:m[0]})}return parts});
  var freqMap=_.countBy(allWords);var freqPairs=_.chain(freqMap).toPairs().sortBy(1).reverse().value();
  var topWords=freqPairs.slice(0,topN).map(function(f){return f[0]});var maxFreq=freqPairs[0]?freqPairs[0][1]:1;
  var ng2=getNg(allWords,2).slice(0,topN),ng3=getNg(allWords,3).slice(0,topN);
  var ws=new Set(topWords);var mx={};topWords.forEach(function(a){mx[a]={};topWords.forEach(function(b){mx[a][b]=0})});
  for(var i=0;i<allWords.length;i++){if(!ws.has(allWords[i]))continue;for(var j=Math.max(0,i-winSize);j<=Math.min(allWords.length-1,i+winSize);j++){if(i!==j&&ws.has(allWords[j]))mx[allWords[i]][allWords[j]]++}}
  var n=topWords.length;var cm=topWords.map(function(_,i){return i});var tw2=0;var wt={};topWords.forEach(function(a){wt[a]=0;topWords.forEach(function(b){var w=mx[a][b]||0;tw2+=w;wt[a]+=w})});tw2/=2;
  if(tw2>0){var imp=true,it=0;while(imp&&it<20){imp=false;it++;for(var ii=0;ii<n;ii++){var bc=cm[ii],bg=0;var ucs=Array.from(new Set(cm));for(var ci=0;ci<ucs.length;ci++){var c=ucs[ci];if(c===cm[ii])continue;var g=0;for(var jj=0;jj<n;jj++){if(cm[jj]!==c)continue;g+=(mx[topWords[ii]][topWords[jj]]||0)-(wt[topWords[ii]]*wt[topWords[jj]])/(2*tw2)}if(g>bg){bg=g;bc=c}}if(bc!==cm[ii]){cm[ii]=bc;imp=true}}}}
  var uComms=Array.from(new Set(cm));var commMap={};topWords.forEach(function(w,i){commMap[w]=uComms.indexOf(cm[i])});
  var relevanceMap=spreadAct(freqMap,eng.syn,eng.pos,wnDepth,decay,flow);var maxRel=Math.max.apply(null,Object.values(relevanceMap).concat([1]));
  var sentCache={};function getSent(lem,p){if(sentCache[lem])return sentCache[lem];var s=eng.sent.ready?eng.sent.score(lem,p):{};sentCache[lem]=s;return s}
  var enriched=renderParas.map(function(parts){return parts.map(function(t){if(t.type!=="word")return t;var s=getSent(t.lemma,t.pos);return Object.assign({},t,{vader:s.vader!=null?s.vader:null,emotions:s.emolex?EMOTIONS.filter(function(e){return s.emolex[e]}):[],vad:s.vad||null,frequency:freqMap[t.lemma]||0,relevance:relevanceMap[t.lemma]||0,community:commMap[t.lemma]!=null?commMap[t.lemma]:null,isTopN:ws.has(t.lemma)})})});
  return{enriched:enriched,freqPairs:freqPairs,freqMap:freqMap,relevanceMap:relevanceMap,maxFreq:maxFreq,maxRel:maxRel,topWords:topWords,commMap:commMap,ng2:ng2,ng3:ng3}}

// ═══ CACHE ═══
function openDB(){return new Promise(function(r,j){var q=indexedDB.open("texturas-cache",1);q.onupgradeneeded=function(){q.result.createObjectStore("assets")};q.onsuccess=function(){r(q.result)};q.onerror=function(){j(q.error)}})}
async function cGet(k){try{var db=await openDB();return new Promise(function(r){var t=db.transaction("assets","readonly");var q=t.objectStore("assets").get(k);q.onsuccess=function(){r(q.result||null)};q.onerror=function(){r(null)}})}catch(e){return null}}
async function cSet(k,v){try{var db=await openDB();return new Promise(function(r){var t=db.transaction("assets","readwrite");t.objectStore("assets").put(v,k);t.oncomplete=function(){r(true)};t.onerror=function(){r(false)}})}catch(e){return false}}
async function loadAsset(key,path,bin,cb){var c=await cGet(key);if(c)return c;var b=ASSET_BASE_URL.endsWith("/")?ASSET_BASE_URL:ASSET_BASE_URL+"/";if(cb)cb("Fetching "+path+"...");try{var r=await fetch(b+path);if(!r.ok)return null;var d=bin?await r.arrayBuffer():await r.json();await cSet(key,d);return d}catch(e){return null}}

// ═══ EXPORT ═══
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}
function csvQ(s){var str=String(s);return str.indexOf(",")!==-1||str.indexOf('"')!==-1||str.indexOf("\n")!==-1?'"'+str.replace(/"/g,'""')+'"':str}
function dlFile(content,filename,type){var b=new Blob([content],{type:type||"text/xml"});var u=URL.createObjectURL(b);var a=document.createElement("a");a.href=u;a.download=filename;a.click();URL.revokeObjectURL(u)}

function genTEI(enriched,docLabel,params,layers,mode){
  var isStandoff=mode==="standoff";var wid=0;var lines=[];var standoff=[];
  lines.push('<?xml version="1.0" encoding="UTF-8"?>');
  lines.push('<TEI xmlns="http://www.tei-c.org/ns/1.0">');
  lines.push('<teiHeader><fileDesc><titleStmt><title>'+esc(docLabel)+'</title></titleStmt>');
  lines.push('<publicationStmt><p>Generated by Texturas v0.7</p></publicationStmt>');
  lines.push('<sourceDesc><p>Multi-layered textual analysis</p></sourceDesc></fileDesc>');
  lines.push('<encodingDesc><classDecl>');
  if(layers.emotion){lines.push('<taxonomy xml:id="emo">');EMOTIONS.forEach(function(e){lines.push('<category xml:id="emo.'+e+'"><catDesc>'+e+'</catDesc></category>')});lines.push('</taxonomy>')}
  if(layers.polarity)lines.push('<taxonomy xml:id="pol"><category xml:id="pol.positive"/><category xml:id="pol.negative"/><category xml:id="pol.neutral"/></taxonomy>');
  lines.push('</classDecl></encodingDesc>');
  lines.push('<profileDesc><textClass><keywords><term>topN='+params.topN+' decay='+params.decay+' flow='+params.flow+'</term></keywords></textClass></profileDesc></teiHeader>');
  lines.push('<text><body>');
  enriched.forEach(function(para,pi){
    lines.push('<p n="'+(pi+1)+'">');
    para.forEach(function(t){
      wid++;var id="w"+wid;
      if(t.type!=="word"){lines.push(esc(t.surface));return}
      if(t.isStop){lines.push('<w xml:id="'+id+'">'+esc(t.surface)+'</w>');return}
      if(isStandoff){lines.push('<w xml:id="'+id+'" lemma="'+esc(t.lemma)+'" pos="'+t.pos+'">'+esc(t.surface)+'</w>');standoff.push({id:id,t:t})}
      else{
        var ana=[];
        if(layers.emotion&&t.emotions&&t.emotions.length)t.emotions.forEach(function(e){ana.push("#emo."+e)});
        if(layers.polarity&&t.vader!==null)ana.push(t.vader>0.05?"#pol.positive":t.vader<-0.05?"#pol.negative":"#pol.neutral");
        if(layers.community&&t.community!==null)ana.push("#comm.C"+(t.community+1));
        var attrs=' xml:id="'+id+'" lemma="'+esc(t.lemma)+'" pos="'+t.pos+'"';
        if(ana.length)attrs+=' ana="'+ana.join(" ")+'"';
        var fs=[];
        if(layers.polarity&&t.vader!==null)fs.push('<f name="vader"><numeric value="'+t.vader.toFixed(4)+'"/></f>');
        if(layers.vad&&t.vad){if(t.vad.v!=null)fs.push('<f name="valence"><numeric value="'+t.vad.v.toFixed(4)+'"/></f>');if(t.vad.a!=null)fs.push('<f name="arousal"><numeric value="'+t.vad.a.toFixed(4)+'"/></f>');if(t.vad.d!=null)fs.push('<f name="dominance"><numeric value="'+t.vad.d.toFixed(4)+'"/></f>')}
        if(layers.frequency)fs.push('<f name="frequency"><numeric value="'+t.frequency+'"/></f>');
        if(layers.relevance)fs.push('<f name="relevance"><numeric value="'+t.relevance.toFixed(2)+'"/></f>');
        lines.push('<w'+attrs+'>'+(fs.length?'<fs>'+fs.join("")+'</fs>':'')+esc(t.surface)+'</w>')}});
    lines.push('</p>')});
  lines.push('</body></text>');
  if(isStandoff&&standoff.length){
    lines.push('<standOff>');
    if(layers.polarity){lines.push('<spanGrp type="polarity">');standoff.forEach(function(d){if(d.t.vader!==null){var pol=d.t.vader>0.05?"positive":d.t.vader<-0.05?"negative":"neutral";lines.push('<span target="#'+d.id+'" ana="#pol.'+pol+'"><fs><f name="vader"><numeric value="'+d.t.vader.toFixed(4)+'"/></f></fs></span>')}});lines.push('</spanGrp>')}
    if(layers.emotion){lines.push('<spanGrp type="emotion">');standoff.forEach(function(d){if(d.t.emotions&&d.t.emotions.length){var a=d.t.emotions.map(function(e){return"#emo."+e}).join(" ");lines.push('<span target="#'+d.id+'" ana="'+a+'"/>')}});lines.push('</spanGrp>')}
    if(layers.vad){lines.push('<spanGrp type="vad">');standoff.forEach(function(d){if(d.t.vad){var fs=[];if(d.t.vad.v!=null)fs.push('<f name="valence"><numeric value="'+d.t.vad.v.toFixed(4)+'"/></f>');if(d.t.vad.a!=null)fs.push('<f name="arousal"><numeric value="'+d.t.vad.a.toFixed(4)+'"/></f>');if(d.t.vad.d!=null)fs.push('<f name="dominance"><numeric value="'+d.t.vad.d.toFixed(4)+'"/></f>');if(fs.length)lines.push('<span target="#'+d.id+'"><fs>'+fs.join("")+'</fs></span>')}});lines.push('</spanGrp>')}
    if(layers.community){lines.push('<spanGrp type="community">');standoff.forEach(function(d){if(d.t.community!==null)lines.push('<span target="#'+d.id+'" ana="#comm.C'+(d.t.community+1)+'"/>')});lines.push('</spanGrp>')}
    lines.push('</standOff>')}
  lines.push('</TEI>');return lines.join("\n")}

function genCorpusTEI(wpd,docs,params,layers,mode){var l=[];l.push('<?xml version="1.0" encoding="UTF-8"?>');l.push('<teiCorpus xmlns="http://www.tei-c.org/ns/1.0">');l.push('<teiHeader><fileDesc><titleStmt><title>Texturas Corpus</title></titleStmt><publicationStmt><p>Texturas v0.7</p></publicationStmt><sourceDesc><p>'+docs.length+' documents</p></sourceDesc></fileDesc></teiHeader>');docs.forEach(function(d){var wr=wpd[d.id];if(!wr)return;var tei=genTEI(wr.enriched,d.label,params,layers,mode);tei=tei.replace('<?xml version="1.0" encoding="UTF-8"?>\n','');l.push(tei)});l.push('</teiCorpus>');return l.join("\n")}

function genCSV(enriched,docLabel,docId,layers){
  var hdr=["doc_id","doc_label","para","position","surface","lemma","pos","is_stop"];
  if(layers.polarity)hdr.push("vader");if(layers.emotion)EMOTIONS.forEach(function(e){hdr.push(e)});if(layers.vad)hdr.push("valence","arousal","dominance");if(layers.frequency)hdr.push("frequency");if(layers.relevance)hdr.push("relevance");if(layers.community)hdr.push("community");
  var rows=[hdr.join(",")];var pos=0;
  enriched.forEach(function(para,pi){para.forEach(function(t){if(t.type!=="word")return;pos++;
    var row=[csvQ(docId),csvQ(docLabel),pi+1,pos,csvQ(t.surface),csvQ(t.lemma),t.pos,t.isStop?1:0];
    if(layers.polarity)row.push(t.vader!==null?t.vader.toFixed(4):"");
    if(layers.emotion)EMOTIONS.forEach(function(e){row.push(t.emotions&&t.emotions.indexOf(e)!==-1?1:0)});
    if(layers.vad){row.push(t.vad&&t.vad.v!=null?t.vad.v.toFixed(4):"");row.push(t.vad&&t.vad.a!=null?t.vad.a.toFixed(4):"");row.push(t.vad&&t.vad.d!=null?t.vad.d.toFixed(4):"")}
    if(layers.frequency)row.push(t.frequency||0);if(layers.relevance)row.push(t.relevance?t.relevance.toFixed(2):"0");if(layers.community)row.push(t.community!==null?"C"+(t.community+1):"");
    rows.push(row.join(","))})});return rows.join("\n")}

function genReport(result,docLabel,params){var l=[];l.push("# Texturas Analysis Report");l.push("**Document:** "+docLabel);l.push("**Generated:** "+new Date().toISOString());l.push("");l.push("## Parameters");l.push("- Top N: "+params.topN+", WN Depth: "+params.wnDepth+", Decay: "+params.decay+", Flow: "+params.flow);l.push("");l.push("## Top Words");result.freqPairs.slice(0,params.topN).forEach(function(p,i){l.push((i+1)+". **"+p[0]+"** freq:"+p[1]+" relev:"+(result.relevanceMap[p[0]]||0).toFixed(1))});l.push("");l.push("## Communities");var comms={};result.topWords.forEach(function(w){var c=result.commMap[w];if(c==null)return;if(!comms[c])comms[c]=[];comms[c].push(w)});Object.keys(comms).forEach(function(c){l.push("- **C"+(parseInt(c)+1)+":** "+comms[c].join(", "))});l.push("");l.push("## Citations");l.push("- NRC EmoLex: Mohammad & Turney (2013)");l.push("- VADER: Hutto & Gilbert (2014)");l.push("- NRC VAD: Mohammad (2018)");l.push("- SentiWordNet: Baccianella et al. (2010)");l.push("- WordNet: Princeton (2010)");l.push("- GloVe: Pennington et al. (2014)");return l.join("\n")}

// ═══ COMPONENTS ═══
function EmoBars({emotions,arousal,showEmo,showAro,enabledSlots}){
  var wrap={position:"absolute",bottom:0,left:0,right:0,height:8,pointerEvents:"none"};
  if(!showEmo)return <span style={wrap}/>;var active=EMOTIONS.filter(function(e){return enabledSlots.has(e)});var present=new Set(emotions);if(!active.length)return <span style={wrap}/>;
  var h=showAro?Math.max(2,Math.round((arousal!=null?arousal:0.5)*8)):6;
  return <span style={Object.assign({},wrap,{display:"flex",gap:0,alignItems:"flex-end",padding:"0 1px"})}>{active.map(function(e){return <span key={e} style={{flex:1,height:present.has(e)?h:0,background:present.has(e)?EC[e]:"transparent",borderRadius:"1px 1px 0 0",opacity:0.85,minWidth:1,maxWidth:4}}/>})}</span>}

function WeaveTooltip({token,x,y}){if(!token||token.isStop)return null;var t=token;
  return <div style={{position:"fixed",left:Math.min(x+12,window.innerWidth-280),top:Math.min(y-10,window.innerHeight-300),zIndex:1000,padding:"10px 14px",background:"#1a1a1aee",border:"1px solid #444",borderRadius:6,fontFamily:"monospace",fontSize:11,color:"#ccc",pointerEvents:"none",maxWidth:260,lineHeight:1.7,backdropFilter:"blur(4px)"}}>
    <div style={{fontSize:13,color:"#4ecdc4",marginBottom:4,fontWeight:"bold"}}>{t.lemma} <span style={{color:"#666",fontWeight:"normal"}}>({t.pos})</span></div>
    {t.vader!==null&&<div>VADER: <span style={{color:t.vader>0.05?"#82e0aa":t.vader<-0.05?"#ff6b6b":"#888"}}>{t.vader>0?"+":""}{t.vader.toFixed(3)}</span></div>}
    {t.emotions.length>0&&<div>Emotions: {t.emotions.map(function(e){return <span key={e} style={{color:EC[e],marginRight:4}}>{e}</span>})}</div>}
    <div>Freq: <span style={{color:"#bb8fce"}}>{t.frequency}</span> {"\u00B7"} Relev: <span style={{color:"#4ecdc4"}}>{t.relevance.toFixed(1)}</span></div>
    {t.community!==null&&<div>Community: <span style={{color:CC[t.community%CC.length]}}>C{t.community+1}</span></div>}
    {t.vad&&<div>V={t.vad.v?t.vad.v.toFixed(2):"?"} A={t.vad.a?t.vad.a.toFixed(2):"?"} D={t.vad.d?t.vad.d.toFixed(2):"?"}</div>}
  </div>}

function WeaveMinimap({enriched,layers,enabledSlots,maxFreq,maxRel,scrollFrac,viewFrac,onSeek,height}){
  var cvRef=useRef();var flatWords=useMemo(function(){var out=[];enriched.forEach(function(para){para.forEach(function(t){if(t.type==="word")out.push(t)});out.push(null)});return out},[enriched]);
  useEffect(function(){var cv=cvRef.current;if(!cv)return;var ctx=cv.getContext("2d");var h=height||400;cv.width=80;cv.height=h;ctx.fillStyle="#0d0d0d";ctx.fillRect(0,0,80,h);var total=flatWords.length;if(!total)return;var rowH=Math.max(1,Math.min(3,h/Math.ceil(total/16)));var cols=16,cw=Math.floor(80/cols);var row=0,col=0;
    flatWords.forEach(function(t){if(!t){row++;col=0;return}var yy=row*rowH,xx=col*cw;if(yy>h)return;var c="#333";if(!t.isStop){if(layers.polarity&&t.vader!==null)c=t.vader>0.05?"#82e0aa":t.vader<-0.05?"#ff6b6b":"#666";else if(layers.community&&t.community!==null)c=CC[t.community%CC.length];else if(layers.emotion){var fe=(t.emotions||[]).filter(function(e){return enabledSlots.has(e)});c=fe.length?EC[fe[0]]:"#555"}else if(layers.relevance){var rn=maxRel>0?Math.log(1+t.relevance)/Math.log(1+maxRel):0;var b=Math.round(50+rn*200);c="rgb("+b+","+Math.round(b*1.2)+","+b+")"}else if(layers.frequency){var fn=maxFreq>0?Math.log(1+t.frequency)/Math.log(1+maxFreq):0;var b2=Math.round(40+fn*180);c="rgb("+b2+","+b2+","+b2+")"}else c="#555"}else c="#222";ctx.fillStyle=c;ctx.fillRect(xx+1,yy,cw-1,Math.max(1,rowH-1));col++;if(col>=cols){col=0;row++}});
    var mapH=(row+1)*rowH;var vpY=scrollFrac*Math.min(mapH,h),vpH=Math.max(8,viewFrac*Math.min(mapH,h));ctx.fillStyle="rgba(78,205,196,0.15)";ctx.fillRect(0,vpY,80,vpH);ctx.strokeStyle="#4ecdc466";ctx.lineWidth=1;ctx.strokeRect(0.5,vpY+0.5,79,vpH-1)},[flatWords,layers,enabledSlots,maxFreq,maxRel,scrollFrac,viewFrac,height]);
  var handleClick=useCallback(function(e){var cv=cvRef.current;if(!cv)return;var r=cv.getBoundingClientRect();onSeek(Math.max(0,Math.min(1,(e.clientY-r.top)/r.height)))},[onSeek]);var dragRef=useRef(false);
  useEffect(function(){var up=function(){dragRef.current=false};window.addEventListener("mouseup",up);return function(){window.removeEventListener("mouseup",up)}},[]);
  return <canvas ref={cvRef} style={{width:80,flexShrink:0,cursor:"pointer",borderLeft:"1px solid #2a2a2a",borderRight:"1px solid #2a2a2a"}} onMouseDown={function(e){dragRef.current=true;handleClick(e)}} onMouseMove={function(e){if(dragRef.current)handleClick(e)}}/>}

function WeaveReader({enriched,layers,highlightLemma,maxFreq,maxRel,onHover,onClick,enabledSlots,showArousal,scrollRef,onScroll,gridSize}){
  if(!enriched||!enriched.length)return <div style={{color:"#555",textAlign:"center",marginTop:60,fontSize:13}}>Run analysis to see annotated text.</div>;
  var gs=gridSize||10,cells=gs*gs;var totalWords=0;enriched.forEach(function(para){para.forEach(function(t){if(t.type==="word"&&t.lower&&t.lower.length>1)totalWords++})});
  var base=Math.floor(totalWords/cells),extra=totalWords%cells;
  var binStarts=useMemo(function(){var s=new Set();var pos=0;for(var i=0;i<cells;i++){s.add(pos);pos+=i<extra?base+1:base}return s},[totalWords,cells,base,extra]);
  function binLabel(wi){var pos=0;for(var i=0;i<cells;i++){var sz=i<extra?base+1:base;if(wi>=pos&&wi<pos+sz)return"["+(Math.floor(i/gs)+1)+","+(i%gs+1)+"]";pos+=sz}return null}
  var wordIdx=0;
  return <div ref={scrollRef} onScroll={onScroll} style={{flex:1,overflowY:"auto",padding:"24px 32px",lineHeight:2.6,wordSpacing:"0.06em"}}>{enriched.map(function(para,pi){return <div key={pi} style={{marginBottom:20}}>{para.map(function(t,ti){if(t.type!=="word")return <span key={ti} style={{fontFamily:"monospace"}}>{t.surface}</span>;var isWord=t.lower&&t.lower.length>1;var wi=isWord?wordIdx:-1;if(isWord)wordIdx++;var showMarker=isWord&&binStarts.has(wi);var marker=showMarker?binLabel(wi):null;
    if(t.isStop)return <span key={ti} style={{fontFamily:"monospace"}}>{showMarker&&<span title={marker} style={{display:"inline-block",width:0,height:"1.1em",borderLeft:"1.5px solid #4ecdc444",marginRight:2,verticalAlign:"middle"}}/>}<span style={{color:"#444"}}>{t.surface}</span></span>;
    var s={fontFamily:"monospace",cursor:"pointer",position:"relative",display:"inline-block",transition:"all 0.15s",padding:"1px 0px",paddingBottom:10,borderRadius:"2px",fontWeight:400};
    s.color=(layers.polarity&&t.vader!==null)?(t.vader>0.05?"#82e0aa":t.vader<-0.05?"#ff6b6b":"#999"):"#ccc";
    if(layers.frequency){var fn=maxFreq>0?Math.log(1+t.frequency)/Math.log(1+maxFreq):0;s.opacity=0.25+fn*0.75}
    if(layers.relevance){var rn=maxRel>0?Math.log(1+t.relevance)/Math.log(1+maxRel):0;s.fontWeight=Math.round(100+rn*600)}
    if(layers.community&&t.community!==null)s.backgroundColor=CC[t.community%CC.length]+"1a";
    if(highlightLemma&&t.lemma===highlightLemma){s.outline="2px solid #4ecdc4";s.outlineOffset="3px";s.borderRadius="3px";s.textShadow="0 0 8px #4ecdc466";if(!s.backgroundColor)s.backgroundColor="#4ecdc40d"}
    var filtEmo=layers.emotion?t.emotions.filter(function(e){return enabledSlots.has(e)}):[];
    return <span key={ti} style={{fontFamily:"monospace"}}>{showMarker&&<span title={marker} style={{display:"inline-block",width:0,height:"1.1em",borderLeft:"1.5px solid #4ecdc466",marginRight:2,verticalAlign:"middle"}}/>}<span style={s} onMouseEnter={function(e){onHover(t,e.clientX,e.clientY)}} onMouseMove={function(e){onHover(t,e.clientX,e.clientY)}} onMouseLeave={function(){onHover(null,0,0)}} onClick={function(){onClick(t.lemma)}}>{t.surface}<EmoBars emotions={filtEmo} arousal={t.vad?t.vad.a:null} showEmo={layers.emotion} showAro={showArousal} enabledSlots={enabledSlots}/></span></span>})}</div>})}</div>}

function WeaveWordPanel({result,topN,highlightLemma,onClickWord,ngMode,setNgMode,sortBy,setSortBy}){
  var isUni=ngMode==="1";var raw=isUni?result.freqPairs.slice(0,topN):ngMode==="2"?result.ng2.slice(0,topN):result.ng3.slice(0,topN);
  var sorted=useMemo(function(){if(!isUni||sortBy!=="relevance")return raw;return raw.slice().sort(function(a,b){return(result.relevanceMap[b[0]]||0)-(result.relevanceMap[a[0]]||0)})},[raw,isUni,sortBy,result.relevanceMap]);
  var maxF=raw[0]?raw[0][1]:1;var maxR=isUni?Math.max.apply(null,raw.map(function(x){return result.relevanceMap[x[0]]||0}).concat([1])):1;
  return <div style={{display:"flex",flexDirection:"column",gap:1,minWidth:0,height:"100%"}}>
    <button onClick={function(){onClickWord(null)}} style={{padding:"4px 8px",marginBottom:2,background:!highlightLemma?"#4ecdc4":"#1a1a1a",color:!highlightLemma?"#111":"#888",border:"1px solid "+(!highlightLemma?"#4ecdc4":"#333"),borderRadius:4,cursor:"pointer",fontSize:11,fontFamily:"monospace",fontWeight:!highlightLemma?"bold":"normal",textAlign:"left"}}>All</button>
    <div style={{display:"flex",gap:0,marginBottom:3,border:"1px solid #333",borderRadius:3,overflow:"hidden"}}>{[["1","1"],["2","2"],["3","3"]].map(function(p){return <button key={p[0]} onClick={function(){setNgMode(p[0]);onClickWord(null)}} style={{flex:1,padding:"3px 0",background:ngMode===p[0]?"#bb8fce":"#1a1a1a",color:ngMode===p[0]?"#111":"#666",border:"none",cursor:"pointer",fontSize:10,fontFamily:"monospace",fontWeight:ngMode===p[0]?"bold":"normal"}}>{p[1]}</button>})}</div>
    {isUni&&<div style={{display:"flex",gap:0,marginBottom:3,border:"1px solid #333",borderRadius:3,overflow:"hidden"}}>{[["freq","Freq"],["relevance","Relev"]].map(function(p){return <button key={p[0]} onClick={function(){setSortBy(p[0])}} style={{flex:1,padding:"3px 0",background:sortBy===p[0]?"#45b7d1":"#1a1a1a",color:sortBy===p[0]?"#111":"#555",border:"none",cursor:"pointer",fontSize:9,fontFamily:"monospace"}}>{p[1]}</button>})}</div>}
    <div style={{flex:1,overflowY:"auto",display:"flex",flexDirection:"column",gap:1}}>{sorted.map(function(pair){var w=pair[0],c=pair[1],rel=isUni?(result.relevanceMap[w]||0):0,isHL=highlightLemma===w;
      return <button key={w} onClick={function(){onClickWord(w===highlightLemma?null:w)}} style={{padding:"3px 8px",background:isHL?"#1a2a2a":"#111",color:isHL?"#ccc":"#aaa",border:"1px solid "+(isHL?"#4ecdc444":"#1a1a1a"),borderRadius:3,cursor:"pointer",fontSize:10,fontFamily:"monospace",textAlign:"left",display:"flex",alignItems:"center",gap:6}}>
        <span style={{flex:1,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",color:isHL?"#4ecdc4":"inherit"}}>{w}</span>
        <div style={{width:46,display:"flex",flexDirection:"column",gap:1,flexShrink:0}}><div style={{height:3,background:"#222",borderRadius:1,overflow:"hidden"}}><div style={{width:(c/maxF*100)+"%",height:"100%",background:"#4ecdc4",borderRadius:1,opacity:isHL?0.9:0.5}}/></div>{isUni&&maxR>0&&<div style={{height:3,background:"#222",borderRadius:1,overflow:"hidden"}}><div style={{width:(rel/maxR*100)+"%",height:"100%",background:"#45b7d1",borderRadius:1,opacity:isHL?0.9:0.5}}/></div>}</div>
        <span style={{width:24,textAlign:"right",fontSize:9,color:isHL?"#888":"#555",flexShrink:0}}>{c}</span></button>})}</div>
    <div style={{display:"flex",gap:8,marginTop:4,fontSize:9,color:"#888"}}><span><span style={{color:"#4ecdc4"}}>{"\u2014"}</span> freq</span>{isUni&&<span><span style={{color:"#45b7d1"}}>{"\u2014"}</span> relev</span>}</div></div>}

function EmoToggle({enabledSlots,setEnabledSlots}){return <div style={{display:"inline-grid",gridTemplateColumns:"repeat(3,10px)",gap:1}}>{EMO_LAYOUT.map(function(item,i){var key=item.emo||"center",on=enabledSlots.has(key),col=item.emo?EC[item.emo]:"#ffffff";return <div key={i} onClick={function(){setEnabledSlots(function(prev){var n=new Set(prev);if(n.has(key))n.delete(key);else n.add(key);return n})}} title={item.emo||"relevance"} style={{width:10,height:10,borderRadius:1,background:on?col:"#555",opacity:on?1:0.5,cursor:"pointer",border:"1px solid "+(on?col+"66":"#444")}}/>})}</div>}

// ═══ MAIN ═══
function WeaveStandalone(){
  var _d=useState([{id:"d1",label:"Document 1",text:""}]);var docs=_d[0],setDocs=_d[1];
  var _ai=useState("d1");var activeInputDoc=_ai[0],setActiveInputDoc=_ai[1];
  var _t=useState("input");var tab=_t[0],setTab=_t[1];
  var _tn=useState(25);var topN=_tn[0],setTopN=_tn[1];
  var _wd=useState(2);var wnDepth=_wd[0],setWnDepth=_wd[1];
  var _dc=useState(0.5);var decay=_dc[0],setDecay=_dc[1];
  var _fl=useState("bi");var flow=_fl[0],setFlow=_fl[1];var winSize=5;
  var _wp=useState({});var weavePerDoc=_wp[0],setWeavePerDoc=_wp[1];
  var _ld=useState(false);var loading=_ld[0],setLoading=_ld[1];
  var _mg=useState("");var msg=_mg[0],setMsg=_mg[1];
  var _wl=useState({polarity:true,emotion:true,frequency:false,relevance:false,community:false});var wLayers=_wl[0],setWLayers=_wl[1];
  var _sa=useState(false);var showArousal=_sa[0],setShowArousal=_sa[1];
  var _es=useState(new Set(EMOTIONS.concat(["center"])));var enabledSlots=_es[0],setEnabledSlots=_es[1];
  var _gs=useState(10);var gridSize=_gs[0],setGridSize=_gs[1];
  var _wh=useState(null);var wHighlight=_wh[0],setWHighlight=_wh[1];
  var _wht=useState(null);var wHovTok=_wht[0],setWHovTok=_wht[1];
  var _whp=useState({x:0,y:0});var wHovPos=_whp[0],setWHovPos=_whp[1];
  var _wng=useState("1");var wNgMode=_wng[0],setWNgMode=_wng[1];
  var _wsb=useState("freq");var wSortBy=_wsb[0],setWSortBy=_wsb[1];
  var _wad=useState(null);var wActiveDoc=_wad[0],setWActiveDoc=_wad[1];
  var _sp=useState(false);var showParams=_sp[0],setShowParams=_sp[1];
  var _el=useState({polarity:true,emotion:true,vad:true,frequency:true,relevance:true,community:true});var expLayers=_el[0],setExpLayers=_el[1];
  var _tm=useState("inline");var teiMode=_tm[0],setTeiMode=_tm[1];
  var readerRef=useRef();var _sf=useState(0);var scrollFrac=_sf[0],setScrollFrac=_sf[1];var _vf=useState(1);var viewFrac=_vf[0],setViewFrac=_vf[1];var _ch=useState(400);var contentH=_ch[0],setContentH=_ch[1];
  var handleReaderScroll=useCallback(function(){var el=readerRef.current;if(!el)return;setScrollFrac(el.scrollHeight>el.clientHeight?el.scrollTop/(el.scrollHeight-el.clientHeight):0);setViewFrac(el.scrollHeight>0?el.clientHeight/el.scrollHeight:1)},[]);
  var handleMinimapSeek=useCallback(function(frac){var el=readerRef.current;if(!el)return;el.scrollTop=frac*(el.scrollHeight-el.clientHeight)},[]);
  var contentRef=useRef();useEffect(function(){if(!contentRef.current)return;var ro=new ResizeObserver(function(en){setContentH(en[0].contentRect.height)});ro.observe(contentRef.current);return function(){ro.disconnect()}},[]);
  var eng=useRef({pos:mkPOS(),lem:mkLem(),syn:mkSyn(),sent:mkSent()});
  useEffect(function(){var cancelled=false;(async function(){var e=eng.current;var pd=await loadAsset("w-p","wordnet/pos-lookup.json",false,setMsg);if(pd&&!cancelled)e.pos.load(pd);var ld=await loadAsset("w-l","wordnet/lemmatizer.json",false,setMsg);if(ld&&!cancelled)e.lem.load(ld);var sd=await loadAsset("w-s","wordnet/synsets.json",false,setMsg);if(sd&&!cancelled)e.syn.load(sd);var el=await loadAsset("l-e","lexicons/nrc-emolex.json",false,setMsg);if(el&&!cancelled)e.sent.lEl(el);var ni=await loadAsset("l-i","lexicons/nrc-intensity.json",false,setMsg);if(ni&&!cancelled)e.sent.lInt(ni);var nv=await loadAsset("l-v","lexicons/nrc-vad.json",false,setMsg);if(nv&&!cancelled)e.sent.lVad(nv);var va=await loadAsset("l-d","lexicons/vader.json",false,setMsg);if(va&&!cancelled)e.sent.lVdr(va);var sw=await loadAsset("l-s","lexicons/sentiwordnet.json",false,setMsg);if(sw&&!cancelled)e.sent.lSwn(sw);if(!cancelled)setMsg("")})();return function(){cancelled=true}},[]);
  var validDocs=docs.filter(function(d){return d.text.trim()});var analyzedIds=Object.keys(weavePerDoc);
  function addDoc(){var id="d"+Date.now();setDocs(function(d){return d.concat([{id:id,label:"Document "+(d.length+1),text:""}])});setActiveInputDoc(id)}
  function rmDoc(id){if(docs.length<=1)return;setDocs(function(d){return d.filter(function(x){return x.id!==id})});if(activeInputDoc===id)setActiveInputDoc(docs[0].id)}
  function updDoc(id,f,v){setDocs(function(d){return d.map(function(x){if(x.id===id){var o={};o[f]=v;return Object.assign({},x,o)}return x})})}
  function handleFiles(files){Array.from(files).forEach(function(f){if(!f.name.endsWith(".txt"))return;var id="d"+Date.now()+"_"+Math.random().toString(36).slice(2,6);var reader=new FileReader();reader.onload=function(ev){setDocs(function(d){return d.filter(function(x){return x.text.trim()}).concat([{id:id,label:f.name.replace(".txt",""),text:ev.target.result}])})};reader.readAsText(f)})}
  function parseSep(text){var parts=text.split(/---DOC(?::?\s*([^-]*))?\s*---/i);var result=[];var pl=null;for(var i=0;i<parts.length;i++){var t=parts[i]?parts[i].trim():"";if(!t)continue;if(i%2===1)pl=t;else{result.push({id:"d"+Date.now()+"_"+i,label:pl||"Document "+(result.length+1),text:t});pl=null}}return result.length>1?result:null}
  var runAnalysis=useCallback(function(){if(!validDocs.length)return;setLoading(true);setMsg("Analyzing...");setTimeout(function(){var e=eng.current,wdr={};validDocs.forEach(function(d){wdr[d.id]=analyzeForWeave(d.text,e,topN,winSize,wnDepth,decay,flow)});setWeavePerDoc(wdr);setWActiveDoc(validDocs[0].id);setWHighlight(null);setLoading(false);setMsg("");setTab("weave");setTimeout(handleReaderScroll,100)},50)},[docs,topN,wnDepth,decay,flow,winSize]);
  function rerunFlow(v){setFlow(v);if(!validDocs.length||!analyzedIds.length)return;setTimeout(function(){var e=eng.current,wdr={};validDocs.forEach(function(d){wdr[d.id]=analyzeForWeave(d.text,e,topN,winSize,wnDepth,decay,v)});setWeavePerDoc(wdr)},50)}
  var rerunTopN=useCallback(function(n){setTopN(n);if(!validDocs.length||!analyzedIds.length)return;setTimeout(function(){var e=eng.current,wdr={};validDocs.forEach(function(d){wdr[d.id]=analyzeForWeave(d.text,e,n,winSize,wnDepth,decay,flow)});setWeavePerDoc(wdr)},50)},[docs,wnDepth,decay,flow,winSize]);
  var rerunDecay=useCallback(function(d){setDecay(d);if(!validDocs.length||!analyzedIds.length)return;setTimeout(function(){var e=eng.current,wdr={};validDocs.forEach(function(doc){wdr[doc.id]=analyzeForWeave(doc.text,e,topN,winSize,wnDepth,d,flow)});setWeavePerDoc(wdr)},50)},[docs,topN,wnDepth,flow,winSize]);
  function toggleWLayer(id){setWLayers(function(prev){var next=Object.assign({},prev);next[id]=!next[id];if(Object.values(next).every(function(v){return!v}))return prev;return next})}
  var activeWR=wActiveDoc?weavePerDoc[wActiveDoc]:null;var curDoc=docs.find(function(d){return d.id===activeInputDoc});var hasMarkers=curDoc&&curDoc.text&&curDoc.text.indexOf("---DOC")!==-1;
  var expParams={topN:topN,wnDepth:wnDepth,decay:decay,flow:flow,winSize:winSize};

  return <div style={{background:"#111",color:"#ddd",height:"100vh",overflow:"hidden",fontFamily:"monospace",display:"flex",flexDirection:"column"}}>
    <div style={{padding:"12px 20px",borderBottom:"1px solid #2a2a2a",display:"flex",alignItems:"center",gap:12}}>
      <span style={{fontSize:18,color:"#4ecdc4",fontWeight:"bold"}}>{"\u2B21"} Texturas</span><span style={{fontSize:11,color:"#555"}}>Weave + Output</span>
      <div style={{marginLeft:"auto",display:"flex",gap:8}}>{eng.current.sent.ready&&<span style={{fontSize:10,color:"#f7dc6f"}}>{"\u25CF"} sent</span>}{eng.current.pos.ready&&<span style={{fontSize:10,color:"#bb8fce"}}>{"\u25CF"} nlp</span>}{eng.current.syn.ready&&<span style={{fontSize:10,color:"#45b7d1"}}>{"\u25CF"} wn</span>}</div></div>
    <div style={{display:"flex",borderBottom:"1px solid #2a2a2a"}}>{[{id:"input",l:"Input"},{id:"weave",l:"Weave"},{id:"output",l:"Output"}].map(function(t){return <button key={t.id} onClick={function(){setTab(t.id)}} style={{padding:"10px 14px",background:tab===t.id?"#1a1a1a":"transparent",color:tab===t.id?"#4ecdc4":"#888",border:"none",borderBottom:tab===t.id?"2px solid #4ecdc4":"2px solid transparent",cursor:"pointer",fontSize:12,fontFamily:"monospace"}}>{t.l}</button>})}<button onClick={function(){setTab("about")}} style={{marginLeft:"auto",padding:"10px 14px",background:tab==="about"?"#1a1a1a":"transparent",color:tab==="about"?"#4ecdc4":"#555",border:"none",borderBottom:tab==="about"?"2px solid #4ecdc4":"2px solid transparent",cursor:"pointer",fontSize:12,fontFamily:"monospace"}}>About</button></div>
    {tab==="weave"&&analyzedIds.length>0&&<div style={{display:"flex",gap:4,padding:"8px 20px",borderBottom:"1px solid #2a2a2a",background:"#151515"}}>{validDocs.filter(function(d){return weavePerDoc[d.id]}).map(function(d){return <button key={d.id} onClick={function(){setWActiveDoc(d.id);setWHighlight(null)}} style={{padding:"4px 12px",borderRadius:3,border:"1px solid "+(wActiveDoc===d.id?"#45b7d1":"#333"),background:wActiveDoc===d.id?"#45b7d1":"#1a1a1a",color:wActiveDoc===d.id?"#111":"#888",fontSize:11,fontFamily:"monospace",cursor:"pointer"}}>{d.label}</button>})}</div>}
    <div style={{flex:1,padding:"16px 20px",overflowY:"auto"}}>

      {/* INPUT */}
      {tab==="input"&&<div style={{maxWidth:800,margin:"0 auto"}}>
        <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}><span style={{fontSize:13,color:"#aaa"}}>Documents ({docs.length})</span><button onClick={addDoc} style={{padding:"4px 12px",background:"#2a2a2a",color:"#4ecdc4",border:"1px solid #444",borderRadius:4,fontSize:12,fontFamily:"monospace",cursor:"pointer"}}>+ Add</button><label style={{padding:"4px 12px",background:"#2a2a2a",color:"#45b7d1",border:"1px solid #444",borderRadius:4,fontSize:12,fontFamily:"monospace",cursor:"pointer"}}>Upload .txt<input type="file" multiple accept=".txt" style={{display:"none"}} onChange={function(ev){handleFiles(ev.target.files)}}/></label></div>
        <div style={{display:"flex",gap:4,marginBottom:12,flexWrap:"wrap"}}>{docs.map(function(d){return <button key={d.id} onClick={function(){setActiveInputDoc(d.id)}} style={{padding:"5px 12px",borderRadius:4,border:"1px solid "+(activeInputDoc===d.id?"#45b7d1":"#333"),background:activeInputDoc===d.id?"#1a2a2a":"#1a1a1a",color:activeInputDoc===d.id?"#45b7d1":"#888",fontSize:12,fontFamily:"monospace",cursor:"pointer"}}>{d.label}{docs.length>1&&<span onClick={function(ev){ev.stopPropagation();rmDoc(d.id)}} style={{marginLeft:8,color:"#666",cursor:"pointer"}}>x</span>}</button>})}</div>
        {docs.filter(function(d){return d.id===activeInputDoc}).map(function(d){return <div key={d.id}><input type="text" value={d.label} onChange={function(ev){updDoc(d.id,"label",ev.target.value)}} style={{width:300,padding:"6px 10px",background:"#1a1a1a",border:"1px solid #444",borderRadius:4,color:"#ccc",fontSize:13,fontFamily:"monospace",boxSizing:"border-box",marginBottom:8}}/><textarea value={d.text} onChange={function(ev){updDoc(d.id,"text",ev.target.value)}} onPaste={function(ev){var t=ev.clipboardData.getData("text");if(t.indexOf("---DOC")!==-1){ev.preventDefault();var p=parseSep(t);if(p){setDocs(p);setActiveInputDoc(p[0].id)}}}} placeholder="Paste text here..." style={{width:"100%",height:"calc(100vh - 380px)",background:"#0d0d0d",border:"1px solid #2a2a2a",borderRadius:6,color:"#ccc",padding:16,fontSize:13,fontFamily:"monospace",resize:"none",boxSizing:"border-box",lineHeight:1.6}}/></div>})}
        <div style={{marginTop:12,display:"flex",gap:8,alignItems:"center"}}><button onClick={runAnalysis} disabled={!validDocs.length||loading} style={{padding:"8px 20px",background:validDocs.length&&!loading?"#4ecdc4":"#333",color:validDocs.length&&!loading?"#111":"#666",border:"none",borderRadius:4,cursor:validDocs.length&&!loading?"pointer":"default",fontFamily:"monospace",fontSize:13,fontWeight:"bold"}}>{"Analyze "+validDocs.length+" doc"+(validDocs.length!==1?"s":"")+" \u2192"}</button>{msg&&<span style={{fontSize:10,color:"#f7dc6f"}}>{msg}</span>}</div></div>}

      {/* WEAVE */}
      {tab==="weave"&&activeWR&&<div style={{maxWidth:1100,margin:"0 auto"}}>
        <div style={{display:"flex",gap:8,marginBottom:12,alignItems:"center",height:36,flexWrap:"wrap"}}>
          {LAYER_CFG.map(function(l){return <button key={l.id} onClick={function(){toggleWLayer(l.id)}} style={{padding:"5px 10px",borderRadius:4,fontSize:11,fontFamily:"monospace",cursor:"pointer",background:wLayers[l.id]?l.color+"22":"#1a1a1a",color:wLayers[l.id]?l.color:"#555",border:"1px solid "+(wLayers[l.id]?l.color:"#333")}}>{l.label}</button>})}
          <div style={{width:40,flexShrink:0}}>{wLayers.emotion&&<EmoToggle enabledSlots={enabledSlots} setEnabledSlots={setEnabledSlots}/>}</div>
          <div style={{width:1,height:22,background:"#333"}}/>
          <div style={{display:"flex",border:"1px solid #333",borderRadius:4,overflow:"hidden"}}>{[10,20,30].map(function(g){return <button key={g} onClick={function(){setGridSize(g)}} style={{padding:"5px 11px",background:gridSize===g?"#bb8fce":"#1a1a1a",color:gridSize===g?"#111":"#666",border:"none",cursor:"pointer",fontSize:11,fontFamily:"monospace",fontWeight:gridSize===g?"bold":"normal"}}>{g+"\u00B2"}</button>})}</div>
          <div style={{marginLeft:"auto",display:"flex",gap:8,alignItems:"center"}}>
            <button onClick={function(){setShowArousal(!showArousal)}} style={{padding:"5px 10px",background:showArousal?"#2a2a1a":"#1a1a1a",color:showArousal?"#f7dc6f":"#555",border:"1px solid "+(showArousal?"#f7dc6f44":"#333"),borderRadius:4,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>Arousal</button>
            <div style={{display:"flex",border:"1px solid #333",borderRadius:4,overflow:"hidden"}}>{[["bi","Bi"],["up","\u2191"],["down","\u2193"]].map(function(p){return <button key={p[0]} onClick={function(){rerunFlow(p[0])}} style={{padding:"5px 9px",background:flow===p[0]?"#45b7d1":"#1a1a1a",color:flow===p[0]?"#111":"#666",border:"none",cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>{p[1]}</button>})}</div>
            <div style={{display:"flex",alignItems:"center",gap:5}}><span style={{fontSize:10,color:"#666"}}>N:</span><input type="range" min={10} max={50} value={topN} onChange={function(ev){rerunTopN(+ev.target.value)}} style={{width:60}}/><span style={{fontSize:10,color:"#aaa",width:16}}>{topN}</span></div>
            <div style={{display:"flex",alignItems:"center",gap:5}}><span style={{fontSize:10,color:"#666"}}>decay:</span><input type="range" min={30} max={80} value={decay*100} onChange={function(ev){rerunDecay(+ev.target.value/100)}} style={{width:50}}/><span style={{fontSize:10,color:"#aaa",width:24}}>{decay.toFixed(2)}</span></div></div></div>
        <div ref={contentRef} style={{display:"flex",gap:10,alignItems:"stretch",height:540,overflow:"hidden"}}>
          <WeaveReader enriched={activeWR.enriched} layers={wLayers} highlightLemma={wHighlight} maxFreq={activeWR.maxFreq} maxRel={activeWR.maxRel} onHover={function(t,x,y){setWHovTok(t);setWHovPos({x:x,y:y})}} onClick={function(lem){setWHighlight(function(prev){return prev===lem?null:lem})}} enabledSlots={enabledSlots} showArousal={showArousal} scrollRef={readerRef} onScroll={handleReaderScroll} gridSize={gridSize}/>
          <WeaveMinimap enriched={activeWR.enriched} layers={wLayers} enabledSlots={enabledSlots} maxFreq={activeWR.maxFreq} maxRel={activeWR.maxRel} scrollFrac={scrollFrac} viewFrac={viewFrac} onSeek={handleMinimapSeek} height={contentH}/>
          <div style={{width:160,flexShrink:0,display:"flex",flexDirection:"column"}}><div style={{fontSize:10,color:"#888",marginBottom:3}}>Top {topN} {"\u00B7"} click to highlight</div><WeaveWordPanel result={activeWR} topN={topN} highlightLemma={wHighlight} onClickWord={function(w){setWHighlight(w)}} ngMode={wNgMode} setNgMode={setWNgMode} sortBy={wSortBy} setSortBy={setWSortBy}/></div></div>
        <div style={{display:"flex",gap:14,marginTop:10,padding:"8px 12px",background:"#0d0d0d",borderRadius:4,border:"1px solid #1a1a1a",fontSize:11,color:"#555",flexWrap:"wrap"}}>
          {wLayers.polarity&&<span><span style={{color:"#82e0aa"}}>{"\u25A0"}</span>/<span style={{color:"#ff6b6b"}}>{"\u25A0"}</span> polarity</span>}
          {wLayers.emotion&&<span><span style={{color:"#f0b27a"}}>{"\u2014"}</span> emotion</span>}
          {wLayers.frequency&&<span><span style={{color:"#bb8fce"}}>{"\u25CB"}</span> brightness</span>}
          {wLayers.relevance&&<span><span style={{color:"#4ecdc4",fontWeight:600}}>B</span> weight</span>}
          {wLayers.community&&<span style={{background:"#4ecdc41a",padding:"0 4px",borderRadius:2}}>community</span>}</div></div>}
      {tab==="weave"&&!activeWR&&<div style={{color:"#555",textAlign:"center",marginTop:60,fontSize:13}}>{"\u2190"} Analyze documents first.</div>}

      {/* OUTPUT */}
      {tab==="output"&&analyzedIds.length>0&&(function(){
        var toggleExp=function(k){setExpLayers(function(prev){var n=Object.assign({},prev);n[k]=!n[k];return n})};
        var layerOpts=[{k:"polarity",l:"VADER polarity",c:"#82e0aa"},{k:"emotion",l:"NRC EmoLex (Plutchik 8)",c:"#f0b27a"},{k:"vad",l:"NRC VAD (V/A/D)",c:"#f7dc6f"},{k:"frequency",l:"Frequency",c:"#bb8fce"},{k:"relevance",l:"Relevance (spreading activation)",c:"#4ecdc4"},{k:"community",l:"Community (Louvain)",c:"#45b7d1"}];
        var aDocs=validDocs.filter(function(d){return weavePerDoc[d.id]});
        return <div style={{maxWidth:800,margin:"0 auto",paddingBottom:40}}>
          <h3 style={{color:"#4ecdc4",fontSize:15,fontWeight:"normal",fontFamily:"monospace",marginBottom:16}}>Export</h3>
          <div style={{marginBottom:24}}>
            <div style={{fontSize:12,color:"#aaa",marginBottom:8}}>Annotation layers:</div>
            <div style={{padding:"2px 6px",background:"#1a1a1a",borderRadius:3,border:"1px solid #333",fontSize:10,color:"#666",marginBottom:8,display:"inline-block"}}>Lemma + POS always included</div>
            {layerOpts.map(function(lo){return <div key={lo.k} onClick={function(){toggleExp(lo.k)}} style={{display:"flex",alignItems:"center",gap:10,padding:"5px 10px",cursor:"pointer",borderRadius:3,marginBottom:1,background:expLayers[lo.k]?"#1a1a1a":"transparent"}}>
              <div style={{width:14,height:14,borderRadius:3,border:"2px solid "+(expLayers[lo.k]?lo.c:"#444"),background:expLayers[lo.k]?lo.c+"33":"transparent",display:"flex",alignItems:"center",justifyContent:"center",fontSize:10,color:lo.c}}>{expLayers[lo.k]?"\u2713":""}</div>
              <span style={{fontSize:11,color:expLayers[lo.k]?"#ccc":"#666"}}>{lo.l}</span></div>})}</div>

          <div style={{marginBottom:24}}>
            <div style={{fontSize:13,color:"#aaa",fontFamily:"monospace",marginBottom:8,borderBottom:"1px solid #333",paddingBottom:6}}>TEI XML</div>
            <div style={{display:"flex",gap:0,marginBottom:10,border:"1px solid #333",borderRadius:4,overflow:"hidden",width:200}}>{[["inline","Inline"],["standoff","Standoff"]].map(function(p){return <button key={p[0]} onClick={function(){setTeiMode(p[0])}} style={{flex:1,padding:"5px 0",background:teiMode===p[0]?"#4ecdc4":"#1a1a1a",color:teiMode===p[0]?"#111":"#666",border:"none",cursor:"pointer",fontSize:10,fontFamily:"monospace"}}>{p[1]}</button>})}</div>
            <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
              {aDocs.map(function(d){return <button key={d.id} onClick={function(){dlFile(genTEI(weavePerDoc[d.id].enriched,d.label,expParams,expLayers,teiMode),d.label.replace(/\s+/g,"_")+"_"+teiMode+".xml")}} style={{padding:"8px 14px",background:"#1a1a1a",border:"1px solid #333",borderRadius:4,cursor:"pointer",fontFamily:"monospace",textAlign:"left"}}><div style={{fontSize:11,color:"#ccc"}}>{d.label}.xml</div><div style={{fontSize:9,color:"#666"}}>{teiMode}</div></button>})}
              {aDocs.length>1&&<button onClick={function(){dlFile(genCorpusTEI(weavePerDoc,aDocs,expParams,expLayers,teiMode),"texturas_corpus_"+teiMode+".xml")}} style={{padding:"8px 14px",background:"#1a2a2a",border:"1px solid #4ecdc444",borderRadius:4,cursor:"pointer",fontFamily:"monospace",textAlign:"left"}}><div style={{fontSize:11,color:"#4ecdc4"}}>Corpus (teiCorpus)</div><div style={{fontSize:9,color:"#666"}}>{teiMode} {"\u00B7"} {aDocs.length} docs</div></button>}</div></div>

          <div style={{marginBottom:24}}>
            <div style={{fontSize:13,color:"#aaa",fontFamily:"monospace",marginBottom:8,borderBottom:"1px solid #333",paddingBottom:6}}>CSV (per-token)</div>
            <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
              {aDocs.map(function(d){return <button key={d.id} onClick={function(){dlFile(genCSV(weavePerDoc[d.id].enriched,d.label,d.id,expLayers),d.label.replace(/\s+/g,"_")+".csv","text/csv")}} style={{padding:"8px 14px",background:"#1a1a1a",border:"1px solid #333",borderRadius:4,cursor:"pointer",fontFamily:"monospace",fontSize:11,color:"#ccc"}}>{d.label}.csv</button>})}
              {aDocs.length>1&&<button onClick={function(){var all=null;aDocs.forEach(function(d){var c=genCSV(weavePerDoc[d.id].enriched,d.label,d.id,expLayers);var lines=c.split("\n");if(!all)all=lines;else all=all.concat(lines.slice(1))});if(all)dlFile(all.join("\n"),"texturas_all.csv","text/csv")}} style={{padding:"8px 14px",background:"#1a2a2a",border:"1px solid #4ecdc444",borderRadius:4,cursor:"pointer",fontFamily:"monospace",fontSize:11,color:"#4ecdc4"}}>All documents.csv</button>}</div></div>

          <div style={{marginBottom:24}}>
            <div style={{fontSize:13,color:"#aaa",fontFamily:"monospace",marginBottom:8,borderBottom:"1px solid #333",paddingBottom:6}}>Summary Report</div>
            <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>{aDocs.map(function(d){return <button key={d.id} onClick={function(){dlFile(genReport(weavePerDoc[d.id],d.label,expParams),d.label.replace(/\s+/g,"_")+"_report.md","text/markdown")}} style={{padding:"8px 14px",background:"#1a1a1a",border:"1px solid #333",borderRadius:4,cursor:"pointer",fontFamily:"monospace",fontSize:11,color:"#ccc"}}>{d.label}_report.md</button>})}</div></div>

          <div style={{padding:10,background:"#1a1a1a",borderRadius:6,border:"1px solid #333",fontFamily:"monospace",fontSize:10,color:"#666",lineHeight:1.7}}>
            <div style={{color:"#aaa",marginBottom:4}}>Notes:</div>
            <div>{"\u2022"} TEI Inline: annotations on &lt;w&gt; + nested &lt;fs&gt;</div>
            <div>{"\u2022"} TEI Standoff: bare &lt;w&gt; + &lt;standOff&gt; &lt;spanGrp&gt; per layer</div>
            <div>{"\u2022"} CSV: one row per token, columns filtered by checkboxes</div>
            <div>{"\u2022"} Layer checkboxes control all export formats</div></div></div>})()}
      {tab==="output"&&analyzedIds.length===0&&<div style={{color:"#555",textAlign:"center",marginTop:60,fontSize:13}}>{"\u2190"} Run analysis before exporting.</div>}

      {/* ABOUT */}
      {tab==="about"&&<div style={{maxWidth:800,margin:"0 auto",lineHeight:1.8,paddingBottom:40}}>
        <h3 style={{color:"#4ecdc4",fontSize:16,fontWeight:"normal",fontFamily:"monospace",marginBottom:6}}>Texturas</h3>
        <p style={{fontSize:11,color:"#666",fontFamily:"monospace",marginBottom:16,fontStyle:"italic"}}>From Latin textura {"\u2014"} weaving. The root of {"\u201C"}text.{"\u201D"} Plural: the many woven layers.</p>
        <p style={{fontSize:12,color:"#999",fontFamily:"monospace",marginBottom:24}}><strong style={{color:"#ccc"}}>Texturas v0.7</strong> {"\u2014"} Multi-layered correlated textual analysis<br/>Ernesto Pe{"\u00F1"}a {"\u00B7"} Northeastern University</p>

        {[{h:"Lexical Resources",items:[
          ["NRC EmoLex","Mohammad & Turney (2013)","Non-commercial research/education"],
          ["NRC Affect Intensity","Mohammad (2018)","Non-commercial research/education"],
          ["NRC VAD","Mohammad (2018)","Non-commercial research/education"],
          ["VADER","Hutto & Gilbert (2014)","MIT"],
          ["SentiWordNet 3.0","Baccianella, Esuli & Sebastiani (2010)","CC BY-SA 4.0"]]},
        {h:"Knowledge Bases",items:[
          ["Princeton WordNet 3.0","Princeton University (2010)","WordNet License (BSD-like)"],
          ["GloVe","Pennington, Socher & Manning (2014)","Public Domain"]]}
        ].map(function(sec){return <div key={sec.h} style={{marginBottom:20}}>
          <h4 style={{color:"#aaa",fontSize:13,fontFamily:"monospace",fontWeight:"normal",marginBottom:8,borderBottom:"1px solid #333",paddingBottom:6}}>{sec.h}</h4>
          {sec.items.map(function(item,i){return <div key={i} style={{padding:8,background:"#1a1a1a",borderRadius:4,border:"1px solid #333",marginBottom:4,fontFamily:"monospace",fontSize:11}}>
            <span style={{color:"#ccc"}}>{item[0]}</span> {"\u00B7"} <span style={{color:"#888"}}>{item[1]}</span> {"\u00B7"} <span style={{color:"#82e0aa"}}>{item[2]}</span></div>})}</div>})}

        <div style={{marginBottom:20}}>
          <h4 style={{color:"#aaa",fontSize:13,fontFamily:"monospace",fontWeight:"normal",marginBottom:8,borderBottom:"1px solid #333",paddingBottom:6}}>Methodology</h4>
          <div style={{padding:8,background:"#1a1a1a",borderRadius:4,border:"1px solid #333",fontFamily:"monospace",fontSize:11}}>
            <span style={{color:"#ccc"}}>Multi-lexicon sentiment analysis</span> {"\u00B7"} <span style={{color:"#888"}}>Mitigates single-source bias per Czarnek & Stillwell (2022)</span></div>
          <div style={{padding:8,background:"#1a1a1a",borderRadius:4,border:"1px solid #333",marginTop:4,fontFamily:"monospace",fontSize:11}}>
            <span style={{color:"#ccc"}}>Spreading activation</span> {"\u00B7"} <span style={{color:"#888"}}>Collins & Loftus (1975), adapted for corpus relevance scoring</span></div>
          <div style={{padding:8,background:"#1a1a1a",borderRadius:4,border:"1px solid #333",marginTop:4,fontFamily:"monospace",fontSize:11}}>
            <span style={{color:"#ccc"}}>Community detection</span> {"\u00B7"} <span style={{color:"#888"}}>Louvain modularity, Blondel et al. (2008)</span></div>
        </div>

        <div style={{padding:10,background:"#1a1a1a",borderRadius:6,border:"1px solid #333",fontFamily:"monospace",fontSize:10,color:"#666",lineHeight:1.7}}>
          <div style={{color:"#f0b27a",marginBottom:4}}>Ethics Note</div>
          <div>Automated sentiment and emotion analysis produces preliminary indicators, not ground truth. Results should be interpreted in context and validated against close reading. Multi-lexicon analysis mitigates but does not eliminate single-source bias.</div>
          <div style={{marginTop:6}}>See: Mohammad (2022), {"\u201C"}Ethics Sheet for Automatic Emotion Recognition and Sentiment Analysis.{"\u201D"}</div>
        </div>

        <div style={{marginTop:20,padding:10,background:"#0d0d0d",borderRadius:4,border:"1px solid #1a1a1a",fontFamily:"monospace",fontSize:10,color:"#444",textAlign:"center"}}>
          Texturas is an open-source research tool {"\u00B7"} Static deployment {"\u00B7"} No data leaves the browser
        </div>
      </div>}
    </div>
    <WeaveTooltip token={wHovTok} x={wHovPos.x} y={wHovPos.y}/>
    {loading&&<div style={{position:"fixed",bottom:20,left:"50%",transform:"translateX(-50%)",padding:"8px 20px",background:"#1a1a1aee",border:"1px solid #444",borderRadius:6,fontSize:11,color:"#f7dc6f",fontFamily:"monospace",zIndex:999}}>{msg||"Processing..."}</div>}
  </div>}
ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(WeaveStandalone));
