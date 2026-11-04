export default function ({data, parentElement, setTriggerValue}) {
  const root = parentElement.querySelector(".timeline");
  const ns = "http://www.w3.org/2000/svg";
  root.replaceChildren();
  const legend = document.createElement("div");
  legend.className = "legend";
  legend.textContent = "Navy: pressure  |  Blue: surface rate  |  Shaded periods: drawdown / buildup";
  root.appendChild(legend);
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 900 410");
  svg.setAttribute("role", "group");
  svg.setAttribute("aria-label", "Designed test rate and pressure timeline");
  root.appendChild(svg);
  const status = document.createElement("p");
  status.className = "status";
  status.setAttribute("role", "status");
  root.appendChild(status);
  const W = 900, left = 90, right = 30, top = 60, bottom = 350;
  const horizon = 1.35 * (data.tp + data.dt_end);
  const x = t => left + t / horizon * (W - left - right);
  const fmt = v => Number(v.toPrecision(5)).toLocaleString("en-US", {maximumFractionDigits: 3});
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const snap = v => Math.round(v * 10) / 10;
  let tp = data.tp, dt = data.dt_end, active = null;
  function el(tag, attrs, text, into = svg) {
    const node = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    into.appendChild(node);
    return node;
  }
  const drawShade = el("rect", {y: top, height: bottom-top, fill: "#EEF2FF"});
  const buildShade = el("rect", {y: top, height: bottom-top, fill: "#EAF6F2"});
  const flowLabel = el("text", {y: 45, fill: "#2251FF", "font-size": 13, "text-anchor": "middle"});
  const buildLabel = el("text", {y: 45, fill: "#0A7D6B", "font-size": 13, "text-anchor": "middle"});
  const curves = el("g", {});
  const pmin = Math.min(...data.pressure), pmax = Math.max(...data.pressure);
  const pad = Math.max((pmax-pmin)*0.08, Math.abs(pmax)*0.001, 0.01);
  const yP = p => 220 - (p - pmin + pad) / (pmax-pmin+2*pad) * 145;
  for (let i=0; i<=4; i++) {
    const p = pmin-pad + i*(pmax-pmin+2*pad)/4, y = yP(p);
    el("line", {x1:left, x2:W-right, y1:y, y2:y, stroke:"#D5DEE9"});
    el("text", {x:left-8, y:y+4, "text-anchor":"end", fill:"#4a5f75"}, fmt(p));
  }
  el("text", {x: 8, y: 18, fill:"#051C2C", "font-weight":600}, `Pressure [${data.pressure_unit}]`);
  el("text", {x: 8, y: 255, fill:"#2251FF", "font-weight":600}, `Rate [${data.rate_unit}]`);
  el("text", {x:left-8, y:294, "text-anchor":"end", fill:"#2251FF"}, fmt(data.rate));
  el("text", {x:left-8, y:344, "text-anchor":"end", fill:"#2251FF"}, "0");
  for (let i=0; i<=6; i++) {
    const t = i*horizon/6, X = x(t);
    el("line", {x1:X, x2:X, y1:top, y2:bottom, stroke:"#D5DEE9"});
    el("text", {x:X, y:373, "text-anchor":"middle", fill:"#4a5f75"}, fmt(t));
  }
  el("text", {x:(left+W-right)/2, y:399, "text-anchor":"middle", fill:"#051C2C"}, "Time since start of flow [hr]");
  const points = data.time.map((t,i) => `${x(t)},${yP(data.pressure[i])}`).join(" ");
  el("polyline", {points, fill:"none", stroke:"#051C2C", "stroke-width":2.5}, undefined, curves);
  const rateLine = el("path", {fill:"none", stroke:"#2251FF", "stroke-width":2.5});
  const handles = ["flow", "end"].map((kind, i) => {
    const group = el("g", {class:"handle", tabindex:0, role:"slider",
      "aria-label":i===0 ? "Start shut-in" : "End test", "aria-valuemin":i===0 ? 1 : 0.1,
      "aria-valuemax":100000, "aria-orientation":"horizontal", "data-handle":kind});
    el("rect", {x:-14, y:top-32, width:28, height:bottom-top+42, fill:"transparent"}, undefined, group);
    el("line", {x1:0, x2:0, y1:top-12, y2:bottom, stroke:i===0 ? "#2251FF" : "#0A7D6B",
      "stroke-width":2, "stroke-dasharray":"5 4"}, undefined, group);
    el("rect", {class:"grip", x:-12, y:top-30, width:24, height:16, rx:3,
      fill:i===0 ? "#2251FF" : "#0A7D6B", stroke:"#fff"}, undefined, group);
    el("title", {}, i===0 ? "Drag to change flowing duration" : "Drag to change shut-in duration", group);
    group.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      event.preventDefault();
      active = kind;
      group.focus();
      group.setPointerCapture(event.pointerId);
    });
    group.addEventListener("pointermove", event => {
      if (active !== kind) return;
      const point = svg.createSVGPoint();
      point.x = event.clientX; point.y = event.clientY;
      const local = point.matrixTransform(svg.getScreenCTM().inverse());
      const t = (local.x-left)/(W-left-right)*horizon;
      if (kind === "flow") tp = clamp(snap(t), 1, 100000);
      else dt = clamp(snap(t-tp), 0.1, 100000);
      update(true);
    });
    group.addEventListener("pointerup", event => {
      if (active !== kind) return;
      active = null;
      group.releasePointerCapture(event.pointerId);
      commit();
    });
    const cancel = () => {
      if (active !== kind) return;
      active = null; tp = data.tp; dt = data.dt_end; update(false);
    };
    group.addEventListener("pointercancel", cancel);
    group.addEventListener("lostpointercapture", cancel);
    group.addEventListener("keydown", event => {
      if (event.key === "Escape") { cancel(); return; }
      if (!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Home","End"].includes(event.key)) return;
      event.preventDefault();
      const step = event.shiftKey ? 10 : 0.1;
      const current = i===0 ? tp : dt;
      const direction = ["ArrowRight","ArrowUp"].includes(event.key) ? 1 : -1;
      const value = event.key === "Home" ? (i===0 ? 1 : 0.1) :
        event.key === "End" ? 100000 : snap(current + direction*step);
      if (i===0) tp = clamp(value, 1, 100000); else dt = clamp(value, 0.1, 100000);
      update(true); commit();
    });
    return group;
  });
  function update(preview) {
    drawShade.setAttribute("x", x(0)); drawShade.setAttribute("width", x(tp)-x(0));
    buildShade.setAttribute("x", x(tp)); buildShade.setAttribute("width", x(tp+dt)-x(tp));
    flowLabel.setAttribute("x", x(tp/2));
    buildLabel.setAttribute("x", x(tp+dt/2));
    flowLabel.textContent = `Drawdown: ${fmt(tp)} hr`;
    buildLabel.textContent = `Buildup: ${fmt(dt)} hr`;
    rateLine.setAttribute("d", `M${x(0)},${data.rate===0 ? 340 : 290} H${x(tp)} V340 H${x(tp+dt)}`);
    handles.forEach((group,i) => {
      group.setAttribute("transform", `translate(${x(i===0 ? tp : tp+dt)},0)`);
      group.setAttribute("aria-valuenow", i===0 ? tp : dt);
      group.setAttribute("aria-valuetext", i===0 ? `${fmt(tp)} hr flowing` : `${fmt(dt)} hr shut-in; test ends at ${fmt(tp+dt)} hr`);
    });
    curves.setAttribute("opacity", preview ? 0.25 : 1);
    status.textContent = `Flow ${fmt(tp)} hr; shut-in ${fmt(dt)} hr; total ${fmt(tp+dt)} hr. ` +
      (preview ? "Release to recompute pressure and diagnostics." : "Drag either handle, or focus it and use arrow keys (0.1 hr; Shift: 10 hr).");
  }
  function commit() {
    if (tp === data.tp && dt === data.dt_end) { update(false); return; }
    status.textContent = "Recomputing the designed test...";
    setTriggerValue("timing", {tp, dt_end:dt, base:[data.tp, data.dt_end]});
  }
  update(false);
}
