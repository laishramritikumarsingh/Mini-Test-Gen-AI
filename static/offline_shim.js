/* Offline mode: answer the app's /api/* calls from data embedded in this HTML file,
   so the page works by double-click with no Python and no internet. */
(() => {
  const D = window.LSE_OFFLINE;
  if (!D) return;
  const json = (obj, status = 200) =>
    Promise.resolve(new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } }));
  const runFor = (b) => D.runs[`${b.seed_a}-${b.seed_b}-${b.method}`];

  window.fetch = (url, opt = {}) => {
    const path = String(url).replace(/^.*?(\/api\/)/, "/api/");
    const body = opt.body ? JSON.parse(opt.body) : {};
    if (path === "/api/status") {
      return json({ status: "ready", message: "Offline page", model_key: "offline", model: D.model, models: [], methods: ["slerp", "lerp", "w_lerp"], max_steps: 9 });
    }
    if (path === "/api/interpolate") {
      const r = runFor(body);
      if (!r) return json({ error: "This offline page only contains the seed pairs in the list." }, 400);
      const { dense, ...rest } = r;
      return json(rest);
    }
    if (path === "/api/frame") {
      const r = runFor(body);
      if (!r) return json({ error: "not available offline" }, 400);
      let best = r.dense[0];
      r.dense.forEach((f) => { if (Math.abs(f.t - body.t) < Math.abs(best.t - body.t)) best = f; });
      return json({ ...best, t: body.t });
    }
    return json({ error: "Not available in the offline page" }, 400);
  };

  // ---- adapt the controls (this script sits at the end of <body>, so the DOM exists) ----
  (() => {
    const $ = (id) => document.getElementById(id);
    $("seedA").value = D.pairs[0][0];
    $("seedB").value = D.pairs[0][1];

    // replace free seed entry with a list of the pre-generated pairs
    const seedRow = document.querySelector(".seed-row");
    const sel = document.createElement("select");
    sel.id = "pairSelect";
    sel.style.width = "100%";
    D.pairs.forEach(([a, b], i) => {
      const o = document.createElement("option");
      o.value = i; o.textContent = `Seed ${a}  →  Seed ${b}`;
      sel.appendChild(o);
    });
    const wrap = document.createElement("div");
    wrap.className = "field";
    wrap.innerHTML = '<label for="pairSelect">Seed pair (pre-generated)</label>';
    wrap.appendChild(sel);
    seedRow.style.display = "none";
    seedRow.parentNode.insertBefore(wrap, seedRow);
    sel.onchange = () => {
      const [a, b] = D.pairs[+sel.value];
      $("seedA").value = a; $("seedB").value = b;
      $("genBtn").click();
    };
    // method buttons regenerate immediately
    document.querySelectorAll("#methodSeg button").forEach((b) =>
      b.addEventListener("click", () => setTimeout(() => $("genBtn").click(), 0)));

    // frames & psi are fixed in the offline file
    $("steps").closest(".field").style.display = "none";
    $("psi").closest(".field").style.display = "none";
    $("exportBtn").style.display = "none";
    const ob = $("offlineBtn"); if (ob) ob.style.display = "none";
    $("modelSelect").style.display = "none";
    $("genBtn").querySelector(".btn-label").textContent = "Show interpolation";
    $("exactBadge").textContent = "25-frame path";

    const note = document.createElement("p");
    note.className = "hint";
    note.innerHTML = `<b>Offline page</b> - every image was generated in advance by the ${D.model.pretrained ? "pretrained" : "UNTRAINED (test)"} StyleGAN2 (${D.model.name}, ψ = ${D.truncation_psi}) on ${D.created}. The slider moves through 25 frames per path. For any seed, run the full app.`;
    wrap.parentNode.insertBefore(note, wrap.nextSibling);
  })();
})();
