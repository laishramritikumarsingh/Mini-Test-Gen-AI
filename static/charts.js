/* Tiny dependency-free SVG chart helpers (works offline). */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const tip = () => document.getElementById("tooltip");

  function el(tag, attrs = {}, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function showTip(evt, html) {
    const t = tip(); t.innerHTML = html; t.classList.remove("hidden");
    const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > innerHeight - 8) y = evt.clientY - h - pad;
    t.style.left = x + "px"; t.style.top = y + "px";
  }
  function hideTip() { tip().classList.add("hidden"); }
  const row = (color, label, val) =>
    `<div class="tt-row">${color ? `<span class="tt-sw" style="background:${color}"></span>` : ""}${label}: <b>${val}</b></div>`;

  function niceTicks(min, max, n = 4) {
    const span = max - min || 1;
    const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function frame(container, W, H, m) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
    return { svg, iw: W - m.l - m.r, ih: H - m.t - m.b, g: el("g", { transform: `translate(${m.l},${m.t})` }, svg) };
  }
  function yAxis(g, ticks, y, iw, fmt) {
    ticks.forEach((v) => {
      el("line", { x1: 0, x2: iw, y1: y(v), y2: y(v), class: "gridline" }, g);
      const t = el("text", { x: -8, y: y(v) + 4, "text-anchor": "end", class: "lbl-text" }, g);
      t.textContent = fmt(v);
    });
  }
  function axisTitle(svg, text, x, y, rotate) {
    const t = el("text", { x, y, class: "lbl-text", "text-anchor": "middle",
      transform: rotate ? `rotate(-90 ${x} ${y})` : "" }, svg);
    t.textContent = text;
  }

  /* ---------- line chart with crosshair tooltip ---------- */
  function lineChart(container, o) {
    const W = 460, H = 230, m = { l: 46, r: 64, t: 12, b: 34 };
    const { svg, g, iw, ih } = frame(container, W, H, m);
    const xs = o.xs;
    const all = o.series.flatMap((s) => s.values).concat(o.ref ? [o.ref.y] : []);
    let [ymin, ymax] = o.yDomain || [Math.min(...all), Math.max(...all)];
    if (!o.yDomain) { const p = (ymax - ymin) * 0.12 || 1; ymin -= p; ymax += p; }
    const x = (v) => ((v - xs[0]) / (xs[xs.length - 1] - xs[0] || 1)) * iw;
    const y = (v) => ih - ((v - ymin) / (ymax - ymin)) * ih;
    const fy = o.yFmt || ((v) => v.toFixed(1));
    yAxis(g, niceTicks(ymin, ymax), y, iw, fy);
    [0, 0.25, 0.5, 0.75, 1].forEach((tv) => {
      const t = el("text", { x: x(tv), y: ih + 18, "text-anchor": "middle", class: "lbl-text" }, g);
      t.textContent = tv;
    });
    el("line", { x1: 0, x2: iw, y1: ih, y2: ih, stroke: css("--muted"), "stroke-width": 1 }, g);
    axisTitle(svg, o.xLabel || "t", m.l + iw / 2, H - 2);

    if (o.ref) {
      el("line", { x1: 0, x2: iw, y1: y(o.ref.y), y2: y(o.ref.y), stroke: css("--muted"), "stroke-width": 1.5, "stroke-dasharray": "5 4" }, g);
      const t = el("text", { x: iw + 6, y: y(o.ref.y) + 4, class: "lbl-text" }, g); t.textContent = o.ref.label;
    }
    o.series.forEach((s) => {
      const d = s.values.map((v, i) => `${i ? "L" : "M"}${x(s.xs ? s.xs[i] : xs[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round",
        "stroke-linecap": "round", "stroke-dasharray": s.dash || "", opacity: s.faded ? 0.45 : 1 }, g);
      if (s.markers) s.values.forEach((v, i) =>
        el("circle", { cx: x(xs[i]), cy: y(v), r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }, g));
      if (o.noDirectLabels) return;
      // direct label at line end
      const last = s.values[s.values.length - 1], lx = x(s.xs ? s.xs[s.xs.length - 1] : xs[xs.length - 1]);
      const lab = el("text", { x: lx + 6, y: y(last) + 4 + (s.labelDy || 0), class: "direct-label" }, g);
      lab.textContent = s.name;
    });
    // current-t marker
    const cur = el("line", { y1: 0, y2: ih, stroke: css("--text"), "stroke-width": 1.5, opacity: 0.7 }, g);
    const setMarker = (t) => { if (t == null) { cur.setAttribute("opacity", 0); return; } cur.setAttribute("x1", x(t)); cur.setAttribute("x2", x(t)); cur.setAttribute("opacity", 0.7); };
    setMarker(o.marker);

    // hover layer
    const hair = el("line", { y1: 0, y2: ih, stroke: css("--muted"), "stroke-width": 1, opacity: 0 }, g);
    const dots = o.series.map((s) => el("circle", { r: 5, fill: s.color, stroke: css("--surface"), "stroke-width": 2, opacity: 0 }, g));
    const hit = el("rect", { x: 0, y: 0, width: iw, height: ih, fill: "transparent" }, g);
    hit.addEventListener("mousemove", (e) => {
      const r = hit.getBoundingClientRect();
      const tv = xs[0] + ((e.clientX - r.left) / r.width) * (xs[xs.length - 1] - xs[0]);
      let html = `<b>t = ${tv.toFixed(3)}</b>`;
      o.series.forEach((s, k) => {
        const sx = s.xs || xs;
        let i = 0, best = Infinity;
        sx.forEach((v, j) => { const dd = Math.abs(v - tv); if (dd < best) { best = dd; i = j; } });
        dots[k].setAttribute("cx", x(sx[i])); dots[k].setAttribute("cy", y(s.values[i])); dots[k].setAttribute("opacity", 1);
        html += row(s.color, s.name, fy(s.values[i]) + (o.yFmtTip ? "" : ""));
      });
      hair.setAttribute("x1", x(tv)); hair.setAttribute("x2", x(tv)); hair.setAttribute("opacity", 1);
      showTip(e, html);
    });
    hit.addEventListener("mouseleave", () => { hair.setAttribute("opacity", 0); dots.forEach((d) => d.setAttribute("opacity", 0)); hideTip(); });
    return { setMarker };
  }

  /* ---------- bar chart ---------- */
  function barChart(container, o) {
    const W = 460, H = 230, m = { l: 46, r: 12, t: 12, b: 34 };
    const { svg, g, iw, ih } = frame(container, W, H, m);
    const n = o.values.length;
    const ymax = (Math.max(...o.values, ...(o.overlay || [0])) || 1) * 1.12;
    const y = (v) => ih - (v / ymax) * ih;
    yAxis(g, niceTicks(0, ymax), y, iw, o.yFmt || ((v) => v.toFixed(0)));
    const slot = iw / n, gap = Math.max(2, slot * 0.18), bw = slot - gap;
    const bars = [];
    o.values.forEach((v, i) => {
      const bx = i * slot + gap / 2, by = y(v), bh = ih - by;
      const r = Math.min(4, bw / 2, bh);
      // rounded top only, anchored to baseline
      const d = bh <= 0 ? "" : `M${bx},${ih}V${by + r}Q${bx},${by} ${bx + r},${by}H${bx + bw - r}Q${bx + bw},${by} ${bx + bw},${by + r}V${ih}Z`;
      const p = el("path", { d, fill: o.color, opacity: o.highlight != null && o.highlight !== i ? 0.55 : 1 }, g);
      bars.push(p);
      const hit = el("rect", { x: i * slot, y: 0, width: slot, height: ih, fill: "transparent" }, g);
      hit.addEventListener("mousemove", (e) => showTip(e, `<b>${o.labels[i]}</b>` + row(o.color, o.name, (o.tipFmt || o.yFmt || ((x) => x.toFixed(2)))(v))
        + (o.overlay ? row(css("--muted"), o.overlayName, o.overlay[i].toFixed(1)) : "")));
      hit.addEventListener("mouseleave", hideTip);
      if (o.onClick) { hit.style.cursor = "pointer"; hit.addEventListener("click", () => o.onClick(i)); }
    });
    if (o.overlay) {
      const d = o.overlay.map((v, i) => `${i ? "L" : "M"}${(i * slot + slot / 2).toFixed(1)},${y(v).toFixed(1)}`).join("");
      el("path", { d, fill: "none", stroke: css("--text-2"), "stroke-width": 2, "stroke-dasharray": "5 4", "pointer-events": "none" }, g);
    }
    el("line", { x1: 0, x2: iw, y1: ih, y2: ih, stroke: css("--muted"), "stroke-width": 1 }, g);
    const every = Math.ceil(n / 8);
    o.labels.forEach((lab, i) => {
      if (i % every && i !== n - 1) return;
      const t = el("text", { x: i * slot + slot / 2, y: ih + 18, "text-anchor": "middle", class: "lbl-text" }, g);
      t.textContent = o.shortLabels ? o.shortLabels[i] : lab;
    });
    axisTitle(svg, o.xLabel || "", m.l + iw / 2, H - 2);
    return { highlight(i) { bars.forEach((b, k) => b.setAttribute("opacity", i == null || i === k ? 1 : 0.55)); } };
  }

  /* ---------- diverging heat strip ---------- */
  function lerpHex(a, b, t) {
    const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
    const ch = (p, s) => (p >> s) & 255;
    const c = [16, 8, 0].map((s) => Math.round(ch(pa, s) + (ch(pb, s) - ch(pa, s)) * t));
    return "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
  }
  function divergeColor(v, lim = 3) {
    const t = Math.max(-1, Math.min(1, v / lim));
    return t < 0 ? lerpHex(css("--mid"), css("--neg"), -t) : lerpHex(css("--mid"), css("--pos"), t);
  }
  function heatStrip(container, rows) {
    const n = rows[0].values.length, W = 460, labW = 70, cell = (W - labW) / n, ch = 44, gap = 8;
    const H = rows.length * (ch + gap) + 20;
    container.innerHTML = "";
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, container);
    const cells = [];
    rows.forEach((r, ri) => {
      const y0 = ri * (ch + gap);
      const t = el("text", { x: 0, y: y0 + ch / 2 + 4, class: "direct-label" }, svg); t.textContent = r.label;
      cells[ri] = [];
      r.values.forEach((v, i) => {
        const rc = el("rect", { x: labW + i * cell, y: y0, width: Math.max(cell - 1, 1), height: ch, rx: 2, fill: divergeColor(v) }, svg);
        rc.addEventListener("mousemove", (e) => showTip(e, `<b>${rows[ri].label}</b>` + row(null, `dim ${i}`, (+rows[ri].values[i]).toFixed(3))));
        rc.addEventListener("mouseleave", hideTip);
        cells[ri].push(rc);
      });
    });
    [0, 12, 24, 36, n - 1].forEach((i) => {
      const t = el("text", { x: labW + i * cell + cell / 2, y: H - 4, "text-anchor": "middle", class: "lbl-text" }, svg);
      t.textContent = i;
    });
    return {
      update(ri, values, label) {
        rows[ri].values = values; if (label) { rows[ri].label = label; svg.querySelectorAll(".direct-label")[ri].textContent = label; }
        values.forEach((v, i) => cells[ri][i].setAttribute("fill", divergeColor(v)));
      },
    };
  }

  function empty(container, msg) { container.innerHTML = `<div class="empty-chart">${msg}</div>`; }

  window.Charts = { lineChart, barChart, heatStrip, empty, css, hideTip };
})();
