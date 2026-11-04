export default function ({data: d, parentElement}) {
  const root = parentElement.querySelector(".schematic");
  if (root._stop) root._stop();
  root.replaceChildren();
  const ns = "http://www.w3.org/2000/svg", p = d.params;
  function html(tag, text, into=root) {
    const n = document.createElement(tag); if (text) n.textContent = text; into.appendChild(n); return n;
  }
  const controls = html("div"); controls.className = "controls";
  const play = html("button", "Play test", controls);
  const reset = html("button", "Restart", controls);
  const speedLabel = html("label", "Speed ", controls);
  const speed = html("select", "", speedLabel); speed.setAttribute("aria-label", "Schematic playback speed");
  for (const v of [0.5, 1, 2, 4]) { const o=html("option", `${v}x`, speed); o.value=v; o.selected=v===1; }
  const scrub = html("input", "", controls);
  Object.assign(scrub, {type:"range", min:"0", max:"1000", step:"1", value:"0"});
  scrub.setAttribute("aria-label", "Schematic test time");
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 760 370"); svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${d.model_label}; ${d.boundary_label}; conceptual reservoir schematic`);
  svg.setAttribute("data-model", d.model); svg.setAttribute("data-boundary", d.boundary);
  root.appendChild(svg);
  function el(tag, attrs, text, into=svg) {
    const n=document.createElementNS(ns, tag);
    for (const [k,v] of Object.entries(attrs)) n.setAttribute(k,v);
    if (text!==undefined) n.textContent=text; into.appendChild(n); return n;
  }
  const navy="#051C2C", blue="#2251FF", teal="#0A7D6B", red="#b3261e";
  const line=(x1,y1,x2,y2,color=navy,w=3,into=svg)=>
    el("line",{x1,y1,x2,y2,stroke:color,"stroke-width":w},undefined,into);
  const label=(x,y,text,color=navy,into=svg)=>el("text",{x,y,fill:color,"font-size":13},text,into);
  el("title",{},`${d.model_label} - ${d.boundary_label}`);
  label(18,25,d.model_label);
  label(18,47,d.boundary_label, "#4a5f75");
  const cross = ["horizontal","limited_entry"].includes(d.model);
  const geometry=el("g",{"data-part":"reservoir"});
  let cx=365, cy=205;
  el("rect",{x:70,y:75,width:620,height:250,fill:"#EEF2F7",stroke:"#D5DEE9"},undefined,geometry);
  if (d.boundary==="rectangle" && !cross) {
    cx=100+550*Math.min(1,Math.max(0,p.xw/p.xe));
    cy=95+210*Math.min(1,Math.max(0,p.yw/p.ye));
  }
  const transient=el("g",{"data-part":"transient"});
  const halo=el("circle",{cx,cy,r:0,fill:blue,"fill-opacity":0.08,stroke:blue,"stroke-dasharray":"5 5"},undefined,transient);
  const recovery=el("circle",{cx,cy,r:0,fill:teal,"fill-opacity":0.14,stroke:teal},undefined,transient);
  const features=el("g",{"data-part":"model-geometry"});
  if (["composite","composite3"].includes(d.model)) {
    const outer=105, inner=d.model==="composite3" ? Math.max(12,outer*Math.min(0.9,p.r_inner/p.r_mid)) : 70;
    if (d.model==="composite3") {
      el("circle",{cx,cy,r:outer,fill:"#D9E7F5","fill-opacity":0.55,stroke:navy},undefined,features);
      label(cx+outer+8,cy-65,"Outer zone",navy,features);
      label(cx+inner+8,cy+55,"Middle zone",navy,features);
    }
    el("circle",{cx,cy,r:inner,fill:"#EAF6F2","fill-opacity":0.7,stroke:teal,"stroke-width":2},undefined,features);
    label(cx-inner+6,cy-10,"Inner zone",teal,features);
    if(d.model==="composite") label(cx+inner+12,cy-25,"Outer zone",navy,features);
  } else if (["dual_pss","dual_slab"].includes(d.model)) {
    for(let i=0;i<6;i++) for(let j=0;j<3;j++) {
      const x=145+i*76,y=100+j*72;
      el("rect",{x,y,width:60,height:53,fill:"#E2E9F0",stroke:"#7A8CA0"},undefined,features);
      if(d.model==="dual_slab") for(let k=1;k<4;k++) line(x,y+k*12,x+60,y+k*12,"#9bb0c8",1,features);
    }
    label(85,345,d.model==="dual_slab" ? "Layered matrix blocks / transient transfer" : "Matrix blocks / pseudo-steady transfer");
    label(460,345,"Connected fracture network",blue);
  } else if (["fracture","fracture_fc"].includes(d.model)) {
    line(cx-145,cy,cx+145,cy,blue,d.model==="fracture_fc" ? 9 : 4,features);
    if(d.model==="fracture_fc") for(let i=-3;i<=3;i++) line(cx+i*35,cy-9,cx+i*35,cy+9,navy,2,features);
    label(cx-145,cy-18,d.model==="fracture_fc" ? "Finite conductivity; FcD controls pressure loss" : "Infinite-conductivity fracture");
    label(cx+85,cy+27,"xf");
  } else if (d.model==="mfhw") {
    const count=Math.max(1,Math.round(p.n_frac));
    const span=Math.max(p.xf,(count-1)*p.f_space);
    const length=500*Math.min(1,(count-1)*p.f_space/span);
    line(cx-length/2,cy,cx+length/2,cy,navy,7,features);
    for(let i=0;i<count;i++) {
      const x=count===1 ? cx : cx-length/2+i*length/(count-1);
      const half=Math.max(15,Math.min(95,250*p.xf/span));
      line(x,cy-half,x,cy+half,blue,4,features);
    }
    label(90,345,`${count} transverse fractures; shared wellbore (plan view)`);
  } else if (cross) {
    const top=95, bottom=305, h=bottom-top;
    const topColor=d.model==="horizontal" && p.top_bc==="const_p" ? teal : red;
    const botColor=d.model==="horizontal" && p.bot_bc==="const_p" ? teal : red;
    line(85,top,675,top,topColor,4,features); line(85,bottom,675,bottom,botColor,4,features);
    label(90,top-8,topColor===teal ? "Constant-pressure top (gas cap)" : "Sealed top");
    label(90,bottom+23,botColor===teal ? "Constant-pressure bottom (aquifer)" : "Sealed bottom");
    if(d.model==="horizontal") {
      const y=top+p.zw_h*h; cy=y;
      halo.setAttribute("cy",cy); recovery.setAttribute("cy",cy);
      line(150,top-22,150,y,navy,6,features); line(150,y,585,y,navy,7,features);
      label(320,y-14,"Horizontal lateral; Lw");
    } else {
      line(cx,top-22,cx,bottom,navy,7,features);
      const y1=top+Math.max(0,p.perf_pos-p.perf_frac/2)*h;
      const y2=top+Math.min(1,p.perf_pos+p.perf_frac/2)*h;
      for(let y=y1;y<=y2;y+=8) line(cx-18,y,cx+18,y,blue,3,features);
      cy=(y1+y2)/2; halo.setAttribute("cy",cy); recovery.setAttribute("cy",cy);
      label(cx+30,cy,"Perforated interval");
    }
    label(510,345,"Cross-section; kv/kh");
  } else {
    label(90,345,"Uniform porous medium; radial flow");
  }
  if (!cross && d.boundary!=="none") {
    const b=el("g",{"data-part":"outer-boundary"});
    const color=["const_p_fault","circle_cp"].includes(d.boundary) ? teal : red;
    if(["circle","circle_cp"].includes(d.boundary)) {
      el("circle",{cx,cy,r:115,fill:"none",stroke:color,"stroke-width":4},undefined,b);
      label(cx+120,cy+85,"re",color,b);
    } else if(d.boundary==="rectangle") {
      el("rect",{x:100,y:95,width:550,height:210,fill:"none",stroke:color,"stroke-width":4},undefined,b);
      label(105,320,"xe x ye; well position xw, yw",color,b);
    } else {
      const norm=Math.max(p.d1,p.d2,1), d1=50+65*p.d1/norm, d2=50+65*p.d2/norm;
      line(cx+d1,85,cx+d1,320,color,5,b);
      label(cx+d1+6,110,"d1",color,b);
      if(d.boundary==="channel") { line(cx-d2,85,cx-d2,320,color,5,b); label(cx-d2+6,110,"d2",color,b); }
      if(d.boundary==="corner") { line(95,cy+d2,650,cy+d2,color,5,b); label(110,cy+d2-8,"d2",color,b); }
    }
  }
  if(cross && d.boundary!=="none") label(90,365,"Lateral boundary applied by solver; not visible in this cross-section",red);
  if(!cross) {
    el("circle",{cx,cy,r:8,fill:navy,stroke:"white","stroke-width":2},undefined,features);
    label(cx+12,cy+22,"Well",navy,features);
  }
  const motion=el("g",{"data-part":"flow-particles"});
  const dots=Array.from({length:12},()=>el("circle",{r:3,fill:blue},undefined,motion));
  const details=html("div"); details.className="parameters";
  for(const text of d.details) html("span",text,details);
  const readout=html("div"); readout.className="readout"; readout.setAttribute("data-part","test-readout");
  html("p","Conceptual geometry, not to scale. Shading and moving dots illustrate drawdown/recovery, not a computed spatial pressure or velocity field.",root).className="note";
  let fraction=0, playing=false, previous=null, frame=null;
  const total=d.tp+d.dt_end;
  function pressure(t) {
    let i=1; while(i<d.time.length-1 && d.time[i]<t) i++;
    const weight=(t-d.time[i-1])/(d.time[i]-d.time[i-1]);
    return d.pressure[i-1]+weight*(d.pressure[i]-d.pressure[i-1]);
  }
  const fmt=v=>Number(v.toPrecision(5)).toLocaleString("en-US",{maximumFractionDigits:3});
  function draw() {
    const t=fraction*total, flowing=t<d.tp, progress=flowing ? t/d.tp : (t-d.tp)/d.dt_end;
    svg.setAttribute("data-phase",flowing ? "drawdown" : "buildup");
    halo.setAttribute("r",d.rate===0 ? 0 : 110*Math.sqrt(Math.min(t/d.tp,1)));
    recovery.setAttribute("r",flowing || d.rate===0 ? 0 : 110*Math.sqrt(progress));
    motion.setAttribute("visibility",flowing && t>0 && d.rate>0 ? "visible" : "hidden");
    dots.forEach((dot,i)=>{
      const angle=i*Math.PI/6, r=20+80*(1-((fraction*16+i/12)%1));
      dot.setAttribute("cx",cx+r*Math.cos(angle)); dot.setAttribute("cy",cy+r*Math.sin(angle));
    });
    scrub.value=Math.round(fraction*1000);
    scrub.setAttribute("aria-valuetext",`${fmt(t)} hr; ${flowing ? "drawdown" : "buildup"}`);
    readout.textContent=`${flowing ? "Drawdown / flowing" : "Buildup / shut-in"} | Time ${fmt(t)} / ${fmt(total)} hr | `+
      `${d.pressure_label} ${fmt(pressure(t))} ${d.pressure_unit} | Surface rate ${fmt(flowing ? d.rate : 0)} ${d.rate_unit}`;
  }
  function stop() {
    playing=false; previous=null; if(frame!==null) cancelAnimationFrame(frame); frame=null; play.textContent="Play test";
  }
  function tick(now) {
    if(!playing || !root.isConnected) { stop(); return; }
    if(previous!==null) fraction=Math.min(1,fraction+(now-previous)/20000*Number(speed.value));
    previous=now; draw();
    if(fraction===1) stop(); else frame=requestAnimationFrame(tick);
  }
  play.onclick=()=>{
    if(playing) { stop(); return; }
    if(fraction===1) fraction=0;
    playing=true; previous=null; play.textContent="Pause test"; frame=requestAnimationFrame(tick);
  };
  reset.onclick=()=>{stop();fraction=0;draw();};
  scrub.oninput=()=>{stop();fraction=Number(scrub.value)/1000;draw();};
  const observer=new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stop();});
  observer.observe(root);
  root._stop=()=>{stop();observer.disconnect();};
  draw();
  return ()=>root._stop();
}
