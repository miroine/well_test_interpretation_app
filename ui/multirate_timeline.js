export default function ({data, parentElement, setTriggerValue}) {
  const root = parentElement.querySelector(".timeline");
  root.replaceChildren();
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 900 550");
  svg.setAttribute("role", "group");
  svg.setAttribute("aria-label", "Multi-rate pressure and rate period editor");
  root.appendChild(svg);
  const status = document.createElement("p");
  status.className = "status";
  status.setAttribute("role", "status");
  root.appendChild(status);
  const left = 95, right = 875, end = data.end ?? data.time[data.time.length - 1];
  const x = t => left + t / end * (right - left);
  const starts = data.schedule.map(s => s[0]);
  const colors = ["#2251FF", "#0A7D6B", "#c2410c", "#7A8CA0"];
  const fmt = v => Number(v.toPrecision(6)).toLocaleString("en-US", {maximumFractionDigits: 4});
  const el = (tag, attrs, text, into = svg) => {
    const n = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k,v]) => n.setAttribute(k,v));
    if (text !== undefined) n.textContent = text;
    into.appendChild(n);
    return n;
  };
  const shade = el("g", {});
  const rates = el("g", {});
  const labels = el("g", {});
  const pmin = data.pressure.reduce((a,b)=>Math.min(a,b), Infinity);
  const pmax = data.pressure.reduce((a,b)=>Math.max(a,b), -Infinity);
  const rmin = Math.min(0, ...data.rates), rmax = Math.max(0, ...data.rates);
  const yp = p => 465 - (p-pmin) / (pmax-pmin || 1) * 145;
  const yr = q => 210 - (q-rmin) / (rmax-rmin || 1) * 125;
  el("text", {x:left,y:30,fill:"#051C2C","font-size":16}, `Rate history [${data.rate_unit}]`);
  el("text", {x:left,y:285,fill:"#051C2C","font-size":16}, `Recorded pressure [${data.pressure_unit}]`);
  for (let i=0;i<=4;i++) {
    const p=pmin+(pmax-pmin)*i/4, q=rmin+(rmax-rmin)*i/4;
    [yp(p),yr(q)].forEach(y=>el("line",{x1:left,x2:right,y1:y,y2:y,stroke:"#D5DEE9"}));
    el("text",{x:left-8,y:yp(p)+4,"text-anchor":"end",fill:"#4a5f75"},fmt(p));
    el("text",{x:left-8,y:yr(q)+4,"text-anchor":"end",fill:"#4a5f75"},fmt(q));
  }
  el("polyline",{points:data.time.map((t,i)=>`${x(t)},${yp(data.pressure[i])}`).join(" "),
    fill:"none",stroke:"#051C2C","stroke-width":2});
  for(let i=0;i<=6;i++) {
    const t=end*i/6;
    el("text",{x:x(t),y:505,"text-anchor":"middle",fill:"#4a5f75"},fmt(t));
  }
  el("text",{x:485,y:535,"text-anchor":"middle",fill:"#051C2C"},"Elapsed test time [hr]");
  const handles = [];
  let active = null;
  const gap = Math.max(end*1e-8,1e-8);
  function update() {
    shade.replaceChildren(); rates.replaceChildren(); labels.replaceChildren();
    starts.forEach((start,i)=>{
      const stop=starts[i+1] ?? end, color=colors[i%colors.length];
      [[60,170],[305,175]].forEach(([y,h])=>{
        el("rect",{x:x(start),y,width:x(stop)-x(start),height:h,fill:color,
          "fill-opacity":i===data.selected ? .17 : .06,stroke:i===data.selected?color:"none"},undefined,shade);
      });
      el("path",{d:`M${x(start)},${yr(data.rates[i])} H${x(stop)}`+
        (i+1<starts.length?` V${yr(data.rates[i+1])}`:""),fill:"none",stroke:"#2251FF","stroke-width":2.5},undefined,rates);
      const text=`P${i+1} ${data.rates[i]===0?"shut-in":"flow"}`;
      el("text",{x:x((start+stop)/2),y:53,"text-anchor":"middle",fill:color,"font-size":11},text,labels);
      el("text",{x:x((start+stop)/2),y:298,"text-anchor":"middle",fill:color,"font-size":11},text,labels);
    });
    handles.forEach(({group,i})=>{
      group.setAttribute("transform",`translate(${x(starts[i])},0)`);
      group.setAttribute("aria-valuenow",starts[i]);
      group.setAttribute("aria-valuemin",starts[i-1]+gap);
      group.setAttribute("aria-valuemax",(starts[i+1]??end)-gap);
      group.setAttribute("aria-valuetext",`${fmt(starts[i])} hours`);
    });
    status.textContent=starts.map((t,i)=>`P${i+1}: ${fmt(t)} hr`).join(" | ");
  }
  function commit(i) {
    if(starts[i]===data.schedule[i][0]) return;
    setTriggerValue("timing",{base:data.schedule,index:i,start:starts[i]});
  }
  starts.forEach((start,i)=>{
    if(i===0) return;
    const group=el("g",{class:"handle",tabindex:0,role:"slider","aria-label":`Start period ${i+1}`,
      "aria-orientation":"horizontal"});
    [[60,230],[305,480]].forEach(([a,b])=>{
      el("rect",{x:-10,y:a-15,width:20,height:b-a+20,fill:"transparent"},undefined,group);
      el("line",{x1:0,x2:0,y1:a,y2:b,stroke:colors[i%colors.length],"stroke-dasharray":"5 4"},undefined,group);
      el("rect",{class:"grip",x:-8,y:a-10,width:16,height:10,rx:2,fill:colors[i%colors.length]},undefined,group);
    });
    handles.push({group,i});
    const move=value=>{starts[i]=Math.max(starts[i-1]+gap,Math.min((starts[i+1]??end)-gap,value));update();};
    group.addEventListener("pointerdown",e=>{
      if(e.button!==0) return;
      e.preventDefault(); active=i; group.focus(); group.setPointerCapture(e.pointerId);
    });
    group.addEventListener("pointermove",e=>{
      if(active!==i) return;
      const pt=svg.createSVGPoint(); pt.x=e.clientX; pt.y=e.clientY;
      const local=pt.matrixTransform(svg.getScreenCTM().inverse());
      move((local.x-left)/(right-left)*end);
    });
    group.addEventListener("pointerup",e=>{
      if(active!==i) return;
      active=null;group.releasePointerCapture(e.pointerId);commit(i);
    });
    const cancel=()=>{if(active===i){active=null;starts[i]=data.schedule[i][0];update();}};
    group.addEventListener("pointercancel",cancel);
    group.addEventListener("lostpointercapture",cancel);
    group.addEventListener("keydown",e=>{
      if(e.key==="Escape"){cancel();return;}
      if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(e.key))return;
      e.preventDefault();
      move(starts[i]+(["ArrowRight","ArrowUp"].includes(e.key)?1:-1)*(e.shiftKey?1:.1));commit(i);
    });
  });
  update();
}
