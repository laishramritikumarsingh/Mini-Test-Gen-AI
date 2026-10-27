/* Latent Space Explorer - frontend logic */
(() => {
  const $ = (id) => document.getElementById(id);
  const C = window.Charts;
  const S = { method: "slerp", run: null, status: null, t: 0, exactTimer: null, reqId: 0, playing: null, charts: {}, dirty: false };

  // ---------------- helpers ----------------
  async function api(path, body) {
    const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
    const r = await fetch(`https://divisions-medal-scroll-pierce.trycloudflare.com${path}`, opt);
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).error || msg; } catch (_) {}
      throw new Error(msg);
    }
    return r;
  }
  function banner(msg, kind = "err") {
    const b = $("banner");
    if (!msg) { b.classList.add("hidden"); return; }
    b.className = "banner " + kind; b.innerHTML = msg;
  }
  function setImg(id, src) {
    const img = $(id); img.src = src; img.parentElement.classList.add("has");
  }
  const params = () => ({
    seed_a: +$("seedA").value, seed_b: +$("seedB").value, steps: +$("steps").value,
    method: S.method, truncation_psi: +$("psi").value,
  });

  // ---------------- model status ----------------
  async function pollStatus() {
    try {
      const st = await (await api("/api/status")).json();
      S.status = st;
      renderStatus(st);
      if (st.status === "ready") { if (!S.run && !S.autoRan) { S.autoRan = true; generate(); } return; }
      if (st.status === "error") return;
    } catch (e) {
      renderStatus({ status: "error", message: "Cannot reach the backend - is app.py still running?" });
    }
    setTimeout(pollStatus, 1500);
  }
  function renderStatus(st) {
    const pill = $("statusPill");
    pill.className = "pill " + (st.status === "ready" ? "pill-ok" : st.status === "error" ? "pill-err" : "pill-wait");
    const m = st.model;
    $("statusText").textContent = st.status === "ready"
      ? `${m.pretrained ? "Pretrained StyleGAN2" : "UNTRAINED demo"} · ${m.resolution}px · ${m.device.toUpperCase()}`
      : st.message || st.status;
    pill.title = st.message || "";
    $("genBtn").disabled = st.status !== "ready";

    if (st.models && !$("modelSelect").options.length) {
      st.models.forEach((mm) => {
        const o = document.createElement("option");
        o.value = mm.key; o.textContent = mm.description + (mm.cached ? "" : " (downloads)");
        $("modelSelect").appendChild(o);
      });
    }
    if (st.model_key && [...$("modelSelect").options].some((o) => o.value === st.model_key)) $("modelSelect").value = st.model_key;

    if (st.status === "error") {
      banner(`<b>Model problem:</b> ${st.message}<br>Check your internet connection for the first-time download, or place a StyleGAN2 <code>.pkl</code> file in the <code>weights/</code> folder and restart. See README → Troubleshooting.`);
    } else if (st.status === "downloading") {
      banner("Downloading the pretrained StyleGAN2 weights (first run only). Progress is shown in the terminal window…", "warn");
    } else if (st.status === "ready" && m && !m.pretrained) {
      banner("Running the <b>untrained</b> demo network (random weights). Images are noise - start without <code>--untrained-demo</code> to use the pretrained model.", "warn");
    } else banner(null);

    if (m) {
      $("pW").textContent = `(${m.num_ws}, ${m.w_dim})`;
      $("pZ").textContent = `(${m.z_dim},)`;
      $("pImg").textContent = `${m.resolution}×${m.resolution}×3`;
      $("modelNote").textContent = `Loaded model: ${m.name} · mapping network ${(m.params_mapping / 1e6).toFixed(2)}M params · synthesis network ${(m.params_synthesis / 1e6).toFixed(2)}M params · ${m.num_ws} style inputs (one w per synthesis layer).`;
    }
  }

  // ---------------- generate ----------------
  const GEN_LABEL = () => S.genLabel || (S.genLabel = $("genBtn").querySelector(".btn-label").textContent);
  async function generate() {
    GEN_LABEL();
    const p = params();
    if (!Number.isInteger(p.seed_a) || !Number.isInteger(p.seed_b) || p.seed_a < 0 || p.seed_b < 0) {
      banner("Seeds must be whole numbers ≥ 0."); return;
    }
    stopPlay();
    const btn = $("genBtn");
    btn.disabled = true; btn.querySelector(".spinner").classList.remove("hidden");
    btn.querySelector(".btn-label").textContent = "Generating…";
    ["wrapA", "wrapB", "wrapCur"].forEach((w) => $(w).classList.add("loading"));
    try {
      const data = await (await api("/api/interpolate", p)).json();
      S.run = data; S.dirty = false;
      banner(null);
      renderRun();
      $("timing").textContent = `${data.frames.length} frames in ${data.elapsed_sec}s`;
    } catch (e) {
      banner("<b>Generation failed:</b> " + e.message);
    } finally {
      btn.disabled = false; btn.querySelector(".spinner").classList.add("hidden");
      btn.querySelector(".btn-label").textContent = S.genLabel;
      ["wrapA", "wrapB", "wrapCur"].forEach((w) => $(w).classList.remove("loading"));
    }
  }

  function renderRun() {
    const r = S.run, f = r.frames, n = f.length;
    setImg("imgA", f[0].image); setImg("imgB", f[n - 1].image);
    $("capA").textContent = r.params.seed_a; $("capB").textContent = r.params.seed_b;
    $("pSeed").textContent = `${r.params.seed_a} → ${r.params.seed_b}`;
    $("pPsi").textContent = `ψ = ${r.params.truncation_psi.toFixed(2)}`;
    $("exportBtn").disabled = false; $("offlineBtn").disabled = false; $("playBtn").disabled = false; $("tSlider").disabled = false;
    const mname = { slerp: "slerp in Z", lerp: "lerp in Z", w_lerp: "lerp in W" }[r.params.method];
    $("stripMeta").textContent = `· ${n} frames · ${mname} · ψ ${r.params.truncation_psi}`;

    // ticks
    $("ticks").innerHTML = f.map((fr, i) => `<span class="${i === 0 || i === n - 1 ? "end" : ""}" style="left:${fr.t * 100}%"></span>`).join("");

    // filmstrip
    $("filmstrip").innerHTML = "";
    f.forEach((fr, i) => {
      const b = document.createElement("button");
      b.className = "frame" + (i === 0 ? " end-a" : i === n - 1 ? " end-b" : "");
      b.title = i === 0 ? "Endpoint A" : i === n - 1 ? "Endpoint B" : `Intermediate frame ${i}`;
      b.innerHTML = `<img src="${fr.image}" alt="frame ${i}"><span class="lbl">${i === 0 ? "A · " : i === n - 1 ? "B · " : ""}t=${fr.t.toFixed(2)}</span>`;
      b.onclick = () => { stopPlay(); setT(fr.t); };
      $("filmstrip").appendChild(b);
    });

    renderCharts();
    renderStats();
    setT(0);
  }

  // ---------------- slider ----------------
  function nearestFrame(t) {
    const f = S.run.frames; let k = 0;
    f.forEach((fr, i) => { if (Math.abs(fr.t - t) < Math.abs(f[k].t - t)) k = i; });
    return k;
  }
  function setT(t, fetchExact = true) {
    if (!S.run) return;
    S.t = t;
    $("tSlider").value = t; $("tVal").textContent = t.toFixed(3);
    const k = nearestFrame(t), fr = S.run.frames[k];
    const onFrame = Math.abs(fr.t - t) < 0.0005;
    setImg("imgCur", fr.image);
    $("exactBadge").classList.add("hidden");
    document.querySelectorAll(".frame").forEach((b, i) => b.classList.toggle("active", i === k));
    updateLatentViews(fr, onFrame ? fr.t : t);
    if (!onFrame && fetchExact) {
      clearTimeout(S.exactTimer);
      S.exactTimer = setTimeout(() => fetchExactFrame(t), 140);
    }
  }
  async function fetchExactFrame(t) {
    const id = ++S.reqId;
    $("wrapCur").classList.add("loading");
    try {
      const d = await (await api("/api/frame", { ...S.run.params, t })).json();
      if (id !== S.reqId || Math.abs(S.t - t) > 1e-6) return;
      setImg("imgCur", d.image);
      $("exactBadge").classList.remove("hidden");
      updateLatentViews(d, t);
    } catch (e) { banner("Frame request failed: " + e.message); }
    finally { if (id === S.reqId) $("wrapCur").classList.remove("loading"); }
  }

  function updateLatentViews(fr, t) {
    const useW = S.run.params.method === "w_lerp";
    const vals = useW ? fr.w_snippet : fr.z_snippet;
    if (S.charts.heat && vals) S.charts.heat.update(1, vals, `${useW ? "w" : "z"}(t=${t.toFixed(2)})`);
    if (S.charts.norm) S.charts.norm.setMarker(t);
    if (S.charts.cos) S.charts.cos.setMarker(t);
    if (S.charts.change) {
      const k = nearestFrame(t);
      S.charts.change.highlight(k === 0 ? null : k - 1);
    }
  }

  // ---------------- play ----------------
  function stopPlay() {
    if (S.playing) { clearInterval(S.playing); S.playing = null; $("playBtn").textContent = "▶"; $("playBtn").title = "Play animation"; }
  }
  function togglePlay() {
    if (S.playing) return stopPlay();
    const f = S.run.frames; let k = nearestFrame(S.t), dir = 1;
    $("playBtn").textContent = "❚❚"; $("playBtn").title = "Pause";
    S.playing = setInterval(() => {
      k += dir;
      if (k >= f.length - 1 || k <= 0) dir = -dir;
      k = Math.max(0, Math.min(f.length - 1, k));
      setT(f[k].t, false);
    }, 160);
  }

  // ---------------- charts ----------------
  function renderCharts() {
    const r = S.run; if (!r) return;
    const c1 = C.css("--accent"), c2 = C.css("--accent-2");
    const useW = r.params.method === "w_lerp";
    const pick = (fr) => (useW ? fr.w_snippet : fr.z_snippet);
    const f = r.frames, n = f.length;

    S.charts.heat = C.heatStrip($("heatmap"), [
      { label: useW ? "w_A" : "z_A", values: pick(f[0]) },
      { label: `${useW ? "w" : "z"}(t=0.00)`, values: pick(f[0]) },
      { label: useW ? "w_B" : "z_B", values: pick(f[n - 1]) },
    ]);
    document.querySelector("#heatmap").previousElementSibling.previousElementSibling.textContent =
      useW ? "Style vector w (first 48 of 512 dims)" : "Latent vector z (first 48 of 512 dims)";

    const nc = r.norm_curve;
    S.charts.norm = C.lineChart($("normChart"), {
      xs: nc.t,
      series: [
        { name: "Slerp", color: c1, values: nc.slerp, faded: r.params.method === "lerp", labelDy: -6 },
        { name: "Lerp", color: c2, values: nc.lerp, faded: r.params.method === "slerp", labelDy: 6 },
      ],
      ref: { y: nc.expected, label: "√512" }, noDirectLabels: true,
      yFmt: (v) => v.toFixed(1), xLabel: "interpolation t", marker: 0,
    });
    $("normChart").insertAdjacentHTML("beforeend",
      `<div class="chart-legend"><span><i style="background:${c1}"></i>Slerp</span><span><i style="background:${c2}"></i>Lerp</span><span><i class="dash"></i>expected radius √512</span>${useW ? "<span>(current run interpolates in W - curves shown for reference)</span>" : ""}</div>`);

    S.charts.cos = C.lineChart($("cosChart"), {
      xs: f.map((x) => x.t),
      series: [
        { name: "to A", color: c1, values: f.map((x) => x.cos_to_a), markers: true, labelDy: 0 },
        { name: "to B", color: c2, values: f.map((x) => x.cos_to_b), markers: true, labelDy: 0 },
      ],
      yFmt: (v) => v.toFixed(2), xLabel: "interpolation t", marker: 0,
    });
    $("cosChart").insertAdjacentHTML("beforeend",
      `<div class="chart-legend"><span><i style="background:${c1}"></i>cosine similarity to A</span><span><i style="background:${c2}"></i>cosine similarity to B</span></div>`);

    if (n > 1) {
      const ch = f.slice(1).map((x) => x.pixel_change);
      S.charts.change = C.barChart($("changeChart"), {
        values: ch, color: c1, name: "pixel change",
        labels: ch.map((_, i) => `frame ${i} → ${i + 1}`), shortLabels: ch.map((_, i) => `${i}→${i + 1}`),
        xLabel: "step between consecutive frames", yFmt: (v) => v.toFixed(0), tipFmt: (v) => v.toFixed(2),
        onClick: (i) => { stopPlay(); setT(f[i + 1].t); },
      });
    }

    // histogram of z_A with expected N(0,1) counts
    const h = r.histograms, bins = h.a.length, [lo, hi] = h.range, bw = (hi - lo) / bins, zdim = r.endpoint_stats.a.dim;
    const cdf = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
    const expected = h.a.map((_, i) => zdim * (cdf(lo + (i + 1) * bw) - cdf(lo + i * bw)));
    const labels = h.a.map((_, i) => `${(lo + i * bw).toFixed(2)} … ${(lo + (i + 1) * bw).toFixed(2)}`);
    C.barChart($("histChart"), {
      values: h.a, color: c1, name: "count in z_A", labels, overlay: expected, overlayName: "expected N(0,1)",
      shortLabels: h.a.map((_, i) => (lo + i * bw).toFixed(0)), xLabel: "value of a latent dimension", tipFmt: (v) => v.toFixed(0),
    });
    $("histChart").insertAdjacentHTML("beforeend",
      `<div class="chart-legend"><span><i style="background:${c1};height:9px"></i>observed counts (z_A, seed ${r.params.seed_a})</span><span><i class="dash"></i>expected for N(0, 1)</span></div>`);

    updateLatentViews(f[nearestFrame(S.t)], S.t);
  }

  function renderStats() {
    const a = S.run.endpoint_stats.a, b = S.run.endpoint_stats.b;
    const rows = [["Dimensions", "dim", 0], ["Mean", "mean", 3], ["Std. deviation", "std", 3], ["Norm ‖z‖", "norm", 2],
      ["Expected norm √d", "expected_norm", 2], ["Min", "min", 2], ["Max", "max", 2]];
    $("statsTable").querySelector("tbody").innerHTML =
      rows.map(([l, k, d]) => `<tr><td>${l}</td><td>${(+a[k]).toFixed(d)}</td><td>${(+b[k]).toFixed(d)}</td></tr>`).join("");
    $("angleNote").innerHTML = `Angle between z_A and z_B: <b>${S.run.angle_deg}°</b> — random high-dimensional Gaussian vectors are almost always close to 90° apart (nearly orthogonal).`;
  }

  function erf(x) { // Abramowitz-Stegun
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  // ---------------- export ----------------
  async function exportZip() {
    const btn = $("exportBtn"); btn.disabled = true; const old = btn.textContent; btn.textContent = "Preparing…";
    try {
      const r = await api("/api/export", {});
      const blob = await r.blob();
      const name = (r.headers.get("Content-Disposition") || "").match(/filename="?([^"]+)"?/);
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = name ? name[1] : "latent_outputs.zip";
      document.body.appendChild(a); a.click(); a.remove();
      banner("Outputs downloaded and also saved in the project's <code>outputs/</code> folder.", "warn");
      setTimeout(() => banner(null), 4000);
    } catch (e) { banner("Export failed: " + e.message); }
    finally { btn.disabled = false; btn.textContent = old; }
  }

  async function saveOffline() {
    const btn = $("offlineBtn"); btn.disabled = true; const old = btn.textContent;
    btn.textContent = "Building page (about a minute)…";
    try {
      const p = { ...params(), seed_a: S.run.params.seed_a, seed_b: S.run.params.seed_b };
      const r = await api("/api/offline_page", p);
      const blob = await r.blob();
      const m = (r.headers.get("Content-Disposition") || "").match(/filename="?([^"]+)"?/);
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = m ? m[1] : "Latent_Space_Explorer_OFFLINE.html";
      document.body.appendChild(a); a.click(); a.remove();
      banner("Offline page downloaded (also saved in <code>outputs/</code>). Copy that one .html file to any laptop and double-click it.", "warn");
    } catch (e) { banner("Offline page failed: " + e.message); }
    finally { btn.disabled = false; btn.textContent = old; }
  }

  // ---------------- wiring ----------------
  function markDirty() {
    if (!S.run) return;
    S.dirty = true; if (!window.LSE_OFFLINE) $("genBtn").querySelector(".btn-label").textContent = "Generate interpolation  ↻";
  }
  document.querySelectorAll(".dice").forEach((b) => b.onclick = () => { $(b.dataset.target).value = Math.floor(Math.random() * 100000); markDirty(); });
  $("swapBtn").onclick = () => { const a = $("seedA").value; $("seedA").value = $("seedB").value; $("seedB").value = a; markDirty(); };
  $("methodSeg").querySelectorAll("button").forEach((b) => b.onclick = () => {
    $("methodSeg").querySelectorAll("button").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-checked", x === b); });
    S.method = b.dataset.v; markDirty();
  });
  $("steps").oninput = () => { $("stepsOut").textContent = $("steps").value; markDirty(); };
  $("psi").oninput = () => { $("psiOut").textContent = (+$("psi").value).toFixed(2); markDirty(); };
  ["seedA", "seedB"].forEach((id) => {
    $(id).oninput = markDirty;
    $(id).onkeydown = (e) => { if (e.key === "Enter" && !$("genBtn").disabled) generate(); };
  });
  $("genBtn").onclick = generate;
  $("exportBtn").onclick = exportZip;
  $("offlineBtn").onclick = saveOffline;
  $("playBtn").onclick = togglePlay;
  $("tSlider").oninput = (e) => { stopPlay(); setT(+e.target.value); };
  $("modelSelect").onchange = async (e) => {
    try {
      await api("/api/load_model", { key: e.target.value });
      S.run = null; S.autoRan = false; $("genBtn").disabled = true;
      renderStatus({ status: "loading", message: "Switching model…" });
      pollStatus();
    } catch (err) { banner(err.message); }
  };
  $("themeBtn").onclick = () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("lse-theme", root.dataset.theme); } catch (_) {}
    renderCharts();
  };
  try { const t = localStorage.getItem("lse-theme"); if (t) document.documentElement.dataset.theme = t; } catch (_) {}
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => renderCharts());

  ["heatmap", "normChart", "cosChart", "changeChart", "histChart"].forEach((id) => C.empty($(id), "Generate an interpolation to see this chart"));
  pollStatus();
})();
