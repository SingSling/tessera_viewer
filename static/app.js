'use strict';
/* TESSERA embedding explorer — browser side.
 * The server sends raw rasters (uint8 PCA / dims, uint16 similarity); all colouring,
 * stretching and thresholding happens here so the controls react instantly. */

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const fmt = (v, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);

let META, W, H, FULL;
const S = {
  year: null,
  mode: 'rgb',
  rgb: { source: 'pca', scope: 'region', pcs: [0, 1, 2], dims: [0, 1, 2], stretch: 'hist', pLo: 2, pHi: 98 },
  view: null, // PCA refitted on a window: {year, r0, c0, h, w, k, comps, explained_variance_ratio, lo, hi, bounds}
  sim: { refs: [], win: 0, lock: true, data: null, key: null, fine: null, n: 0, cmap: 'viridis', lo: 0, hi: 1, thr: -1, seq: 0 },
  shown: null, // what the canvas currently holds, for the hover readout
};

/* ---------------- data access ---------------- */
const cache = new Map();
async function fetchOk(url) {
  const r = await fetch(url);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).error; } catch (_) { /* not json */ }
    throw new Error(msg);
  }
  return r;
}
function band(kind, year, i) {
  const key = `${kind}:${year}:${i}`;
  if (!cache.has(key)) {
    const p = fetchOk(`/api/${kind}?year=${year}&${kind === 'pca' ? 'k' : 'd'}=${i}`)
      .then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b));
    p.catch(() => cache.delete(key));
    cache.set(key, p);
    if (cache.size > 48) cache.delete(cache.keys().next().value);
  }
  return cache.get(key);
}

let statusTimer;
function status(msg, isErr = false) {
  const el = $('#status');
  clearTimeout(statusTimer);
  if (!msg) { el.hidden = true; return; }
  el.textContent = msg;
  el.classList.toggle('err', isErr);
  el.hidden = false;
  if (isErr) statusTimer = setTimeout(() => (el.hidden = true), 5000);
}

/* ---------------- map ---------------- */
const map = L.map('map', { zoomSnap: 0.25, zoomDelta: 0.5 });
const basemaps = {
  sat: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 20, maxNativeZoom: 19, attribution: 'Imagery © Esri, Maxar, Earthstar Geographics' }),
  osm: L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 20, maxNativeZoom: 19, attribution: '© OpenStreetMap contributors' }),
  light: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 20, maxNativeZoom: 16, attribution: 'Tiles © Esri, HERE, Garmin, © OpenStreetMap contributors' }),
};
let basemap = basemaps.sat.addTo(map);
map.createPane('labels');
map.getPane('labels').style.zIndex = 450;
map.getPane('labels').style.pointerEvents = 'none';
const labels = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', {
  pane: 'labels', maxZoom: 20, maxNativeZoom: 19 }).addTo(map);

// An ImageOverlay whose element is a <canvas>, so it can be repainted without PNG encoding.
const CanvasOverlay = L.ImageOverlay.extend({
  _initImage() {
    const el = (this._image = this._url);
    L.DomUtil.addClass(el, 'leaflet-image-layer');
    if (this._zoomAnimated) L.DomUtil.addClass(el, 'leaflet-zoom-animated');
    if (this.options.className) L.DomUtil.addClass(el, this.options.className);
    el.onselectstart = L.Util.falseFn;
    el.onmousemove = L.Util.falseFn;
  },
});
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
let overlay, imgData;
const refLayer = L.layerGroup();

// Grid <-> map coordinates (data grid is Web Mercator, top-left origin).
function toPixel(latlng) {
  const p = L.CRS.EPSG3857.project(latlng);
  return { r: Math.floor((META.y0 - p.y) / META.res), c: Math.floor((p.x - META.x0) / META.res) };
}
function unproj(x, y) { return L.CRS.EPSG3857.unproject(L.point(x, y)); }
function pixelCenter(r, c) { return unproj(META.x0 + (c + 0.5) * META.res, META.y0 - (r + 0.5) * META.res); }
function windowBounds(r0, c0, h, w) {
  return L.latLngBounds(unproj(META.x0 + c0 * META.res, META.y0 - (r0 + h) * META.res),
                        unproj(META.x0 + (c0 + w) * META.res, META.y0 - r0 * META.res));
}
const inside = (r, c) => r >= 0 && c >= 0 && r < H && c < W;

function paint(w, h, bounds, fill) {
  if (canvas.width !== w || canvas.height !== h || !imgData) {
    canvas.width = w; canvas.height = h;
    imgData = ctx.createImageData(w, h);
  }
  fill(new Uint32Array(imgData.data.buffer));
  ctx.putImageData(imgData, 0, 0);
  overlay.setBounds(bounds);
}
function clearCanvas() { ctx.clearRect(0, 0, canvas.width, canvas.height); S.shown = null; }

/* ---------------- RGB composite ---------------- */
const histCache = new WeakMap();
function hist256(a) {
  if (histCache.has(a)) return histCache.get(a);
  const h = new Float64Array(256);
  for (let i = 0; i < a.length; i++) h[a[i]]++;
  h[0] = 0; // no data
  histCache.set(a, h);
  return h;
}
function stretchLUT(a) {
  const h = hist256(a);
  let n = 0, s1 = 0, s2 = 0;
  for (let v = 1; v < 256; v++) { n += h[v]; s1 += v * h[v]; s2 += v * v * h[v]; }
  const lut = new Uint8ClampedArray(256);
  if (!n) return lut;
  const mode = S.rgb.stretch;
  if (mode === 'hist') {
    let cum = 0;
    for (let v = 1; v < 256; v++) { lut[v] = Math.round(255 * (cum + h[v] / 2) / n); cum += h[v]; }
    return lut;
  }
  let lo, hi;
  if (mode === 'std') {
    const m = s1 / n, sd = Math.sqrt(Math.max(s2 / n - m * m, 1e-9));
    lo = m - 2.5 * sd; hi = m + 2.5 * sd;
  } else {
    const at = (p) => { let cum = 0; const t = n * p / 100; for (let v = 1; v < 256; v++) { cum += h[v]; if (cum >= t) return v; } return 255; };
    lo = at(S.rgb.pLo); hi = at(S.rgb.pHi);
  }
  for (let v = 1; v < 256; v++) lut[v] = 255 * (v - lo) / Math.max(hi - lo, 1);
  return lut;
}

let rgbSeq = 0;
async function renderRGB() {
  const seq = ++rgbSeq;
  const year = S.year;
  let chans, w = W, h = H, r0 = 0, c0 = 0, bounds = FULL, ranges = null, labelsTxt;
  try {
    if (S.rgb.source === 'dims') {
      status('Loading dimensions…');
      chans = await Promise.all(S.rgb.dims.map((d) => band('dim', year, d)));
      labelsTxt = S.rgb.dims.map((d) => `dim ${d}`);
    } else if (S.rgb.scope === 'view' && S.view) {
      const v = S.view;
      chans = S.rgb.pcs.map((k) => v.comps[Math.min(k, v.k - 1)]);
      ({ w, h, r0, c0, bounds } = v);
      ranges = S.rgb.pcs.map((k) => [v.lo[k], v.hi[k]]);
      labelsTxt = S.rgb.pcs.map((k) => `PC${k + 1}`);
    } else {
      status('Loading components…');
      chans = await Promise.all(S.rgb.pcs.map((k) => band('pca', year, k)));
      const p = META.pca[year];
      ranges = S.rgb.pcs.map((k) => [p.lo[k], p.hi[k]]);
      labelsTxt = S.rgb.pcs.map((k) => `PC${k + 1}`);
    }
  } catch (e) { status(e.message, true); return; }
  status(null);
  if (seq !== rgbSeq || S.mode !== 'rgb') return;
  const [R, G, B] = chans;
  const [lr, lg, lb] = chans.map(stretchLUT);
  paint(w, h, bounds, (d32) => {
    for (let i = 0; i < d32.length; i++) {
      const r = R[i];
      d32[i] = r ? (0xff000000 | (lb[B[i]] << 16) | (lg[G[i]] << 8) | lr[r]) : 0;
    }
  });
  S.shown = { kind: 'rgb', chans, w, h, r0, c0, ranges, labels: labelsTxt };
}

function evList() {
  if (S.rgb.scope === 'view' && S.view) return S.view.explained_variance_ratio;
  return META.pca[S.year].explained_variance_ratio;
}
function buildChannelOptions() {
  const ev = evList();
  ['r', 'g', 'b'].forEach((ch, i) => {
    const sel = $(`#ch-${ch}`);
    sel.innerHTML = ev.map((e, k) => `<option value="${k}">PC${k + 1} · ${(e * 100).toFixed(1)}%</option>`).join('');
    S.rgb.pcs[i] = Math.min(S.rgb.pcs[i], ev.length - 1);
    sel.value = S.rgb.pcs[i];
    $(`#dim-${ch}`).value = S.rgb.dims[i];
  });
  renderEV();
}
function renderEV() {
  const el = $('#ev');
  if (S.rgb.source !== 'pca') { el.innerHTML = ''; return; }
  const ev = evList();
  const max = Math.max(...ev);
  const cls = (k) => ['r', 'g', 'b'].filter((_, i) => S.rgb.pcs[i] === k);
  el.innerHTML = ev.map((e, k) => {
    const c = cls(k);
    return `<span>PC${k + 1}</span><div class="bar ${c.length ? 'on ' + c[0] : ''}" style="width:${(e / max) * 100}%"></div><span class="pct">${(e * 100).toFixed(1)}%</span>`;
  }).join('');
}

async function loadView(r0, r1, c0, c1) {
  status('Fitting PCA to this view…');
  try {
    const r = await fetchOk(`/api/viewpca?year=${S.year}&r0=${r0}&r1=${r1}&c0=${c0}&c1=${c1}`);
    const info = JSON.parse(r.headers.get('X-Info'));
    const buf = new Uint8Array(await r.arrayBuffer());
    const n = info.h * info.w;
    info.comps = Array.from({ length: info.k }, (_, i) => buf.subarray(i * n, (i + 1) * n));
    info.bounds = windowBounds(info.r0, info.c0, info.h, info.w);
    info.year = S.year;
    S.view = info;
    status(null);
    return true;
  } catch (e) { status(e.message, true); return false; }
}
async function fitToView() {
  const b = map.getBounds();
  const a = toPixel(b.getNorthWest()), z = toPixel(b.getSouthEast());
  const r0 = clamp(a.r, 0, H), r1 = clamp(z.r + 1, 0, H), c0 = clamp(a.c, 0, W), c1 = clamp(z.c + 1, 0, W);
  if (r1 - r0 < 8 || c1 - c0 < 8) { status('Move the map over the data first', true); return false; }
  return loadView(r0, r1, c0, c1);
}

/* ---------------- similarity ---------------- */
const FINE = 4000; // histogram bins over [-1, 1]
const dec = (v) => ((v - 1) / 65534) * 2 - 1; // uint16 -> similarity
const binOf = (s) => clamp(Math.floor(((s + 1) / 2) * FINE), 0, FINE - 1);
const binVal = (b) => (b / FINE) * 2 - 1;

function simKey() {
  return `${S.year}|${S.sim.win}|${S.sim.refs.map((p) => `${p.r}:${p.c}:${S.sim.lock ? p.year : S.year}`).join(',')}`;
}
async function computeSim() {
  drawRefs();
  renderRefChips();
  if (!S.sim.refs.length) {
    S.sim.data = null; S.sim.key = null;
    clearCanvas(); drawHist(); updateStats();
    return;
  }
  const key = simKey();
  if (key === S.sim.key && S.sim.data) { renderSim(); return; }
  const seq = ++S.sim.seq;
  const refs = S.sim.refs.map((p) => `${p.r}:${p.c}:${S.sim.lock ? p.year : S.year}`).join(',');
  status('Computing similarity…');
  try {
    const t0 = performance.now();
    const r = await fetchOk(`/api/sim?year=${S.year}&win=${S.sim.win}&refs=${refs}`);
    const buf = await r.arrayBuffer();
    if (seq !== S.sim.seq) return;
    S.sim.data = new Uint16Array(buf);
    S.sim.key = key;
    S.sim.nref = +r.headers.get('X-Ref-Pixels');
    S.sim.ms = performance.now() - t0;
    buildFineHist();
    autoRange();
    status(null);
    if (S.mode === 'sim') renderSim();
  } catch (e) { status(e.message, true); }
}
function buildFineHist() {
  const d = S.sim.data, h = new Float64Array(FINE);
  let n = 0;
  for (let i = 0; i < d.length; i++) {
    const v = d[i];
    if (v) { h[binOf(dec(v))]++; n++; }
  }
  S.sim.fine = h; S.sim.n = n;
}
function quantile(p) {
  const h = S.sim.fine, t = S.sim.n * p;
  let cum = 0;
  for (let b = 0; b < FINE; b++) { cum += h[b]; if (cum >= t) return binVal(b + 1); }
  return 1;
}
function autoRange() {
  if (!S.sim.fine) return;
  S.sim.lo = Math.round(quantile(0.02) * 200) / 200;
  S.sim.hi = Math.round(Math.min(1, quantile(0.9995)) * 200) / 200;
  if (S.sim.hi <= S.sim.lo) S.sim.hi = Math.min(1, S.sim.lo + 0.05);
  syncSliders();
}
function syncSliders() {
  $('#lo').value = S.sim.lo; $('#hi').value = S.sim.hi; $('#thr').value = S.sim.thr;
  $('#lo-v').textContent = S.sim.lo.toFixed(2);
  $('#hi-v').textContent = S.sim.hi.toFixed(2);
  $('#thr-v').textContent = S.sim.thr <= -1 ? 'off' : S.sim.thr.toFixed(2);
}
function cmapRGB(t) { // t in [0,1] -> [r,g,b]
  const m = META.colormaps[S.sim.cmap], i = Math.round(clamp(t, 0, 1) * 255) * 3;
  return [m[i], m[i + 1], m[i + 2]];
}
function renderSim() {
  const d = S.sim.data;
  if (!d) { clearCanvas(); drawHist(); updateStats(); return; }
  const { lo, hi, thr } = S.sim;
  const lut = new Uint32Array(65536);
  for (let v = 1; v < 65536; v++) {
    const s = dec(v);
    if (s < thr) continue;
    const [r, g, b] = cmapRGB((s - lo) / (hi - lo || 1e-6));
    lut[v] = 0xff000000 | (b << 16) | (g << 8) | r;
  }
  paint(W, H, FULL, (d32) => { for (let i = 0; i < d32.length; i++) d32[i] = lut[d[i]]; });
  S.shown = { kind: 'sim' };
  drawHist();
  updateStats();
}

// Histogram of similarity, bars coloured with the map's colour scale.
let histBins = null;
function drawHist() {
  const cv = $('#hist'), dpr = window.devicePixelRatio || 1;
  const cw = cv.clientWidth, ch = cv.clientHeight;
  cv.width = cw * dpr; cv.height = ch * dpr;
  const g = cv.getContext('2d');
  g.scale(dpr, dpr);
  g.clearRect(0, 0, cw, ch);
  const css = getComputedStyle(document.documentElement);
  if (!S.sim.fine) {
    g.fillStyle = css.getPropertyValue('--muted');
    g.font = '12px system-ui';
    g.fillText('Pick a pixel on the map', 8, ch / 2);
    $('#ax-lo').textContent = ''; $('#ax-hi').textContent = '';
    histBins = null;
    return;
  }
  const xmin = Math.max(-1, Math.floor(quantile(0.002) * 20) / 20), xmax = 1;
  const nb = 90, step = (xmax - xmin) / nb, bins = new Float64Array(nb);
  const f = S.sim.fine;
  for (let b = 0; b < FINE; b++) {
    if (!f[b]) continue;
    const s = binVal(b) + 1 / FINE;
    if (s < xmin) continue;
    bins[Math.min(nb - 1, Math.floor((s - xmin) / step))] += f[b];
  }
  const max = Math.sqrt(Math.max(...bins)) || 1;
  const bw = cw / nb, gap = bw > 3 ? 1 : 0;
  const muted = css.getPropertyValue('--line');
  for (let i = 0; i < nb; i++) {
    if (!bins[i]) continue;
    const s = xmin + (i + 0.5) * step;
    const bh = Math.max(1, (Math.sqrt(bins[i]) / max) * (ch - 4));
    const [r, gg, b] = cmapRGB((s - S.sim.lo) / (S.sim.hi - S.sim.lo || 1e-6));
    g.fillStyle = s < S.sim.thr ? muted : `rgb(${r},${gg},${b})`;
    g.fillRect(i * bw, ch - bh, bw - gap, bh);
  }
  if (S.sim.thr > xmin) {
    const x = ((S.sim.thr - xmin) / (xmax - xmin)) * cw;
    g.fillStyle = css.getPropertyValue('--ink');
    g.fillRect(Math.round(x) - 1, 0, 2, ch);
  }
  $('#ax-lo').textContent = xmin.toFixed(2);
  $('#ax-hi').textContent = xmax.toFixed(2);
  histBins = { bins, xmin, step, nb };
}
$('#hist').addEventListener('mousemove', (e) => {
  const tip = $('#hist-tip');
  if (!histBins) { tip.hidden = true; return; }
  const rect = e.currentTarget.getBoundingClientRect();
  const i = clamp(Math.floor(((e.clientX - rect.left) / rect.width) * histBins.nb), 0, histBins.nb - 1);
  const a = histBins.xmin + i * histBins.step, n = histBins.bins[i];
  tip.innerHTML = `${a.toFixed(3)}–${(a + histBins.step).toFixed(3)} · ${((n / S.sim.n) * 100).toFixed(2)}% · ${fmtArea(n)}`;
  tip.style.left = `${clamp(e.clientX - rect.left, 70, rect.width - 70)}px`;
  tip.hidden = false;
});
$('#hist').addEventListener('mouseleave', () => ($('#hist-tip').hidden = true));
$('#hist').addEventListener('click', (e) => { // click on the histogram = set "hide below"
  if (!histBins) return;
  const rect = e.currentTarget.getBoundingClientRect();
  S.sim.thr = Math.round((histBins.xmin + ((e.clientX - rect.left) / rect.width) * histBins.nb * histBins.step) * 200) / 200;
  syncSliders(); scheduleSim();
});

function pixelArea() { // ground area of one pixel in ha
  const lat = (META.bbox[1] + META.bbox[3]) / 2;
  const side = META.res * Math.cos((lat * Math.PI) / 180);
  return (side * side) / 1e4;
}
function fmtArea(npx) {
  const ha = npx * pixelArea();
  return ha >= 100 ? `${(ha / 100).toFixed(1)} km²` : `${ha.toFixed(1)} ha`;
}
function updateStats() {
  const el = $('#sim-stats');
  if (!S.sim.fine) { el.textContent = ''; return; }
  const parts = [`median ${quantile(0.5).toFixed(2)}`];
  if (S.sim.thr > -1) {
    let above = 0;
    for (let b = binOf(S.sim.thr); b < FINE; b++) above += S.sim.fine[b];
    parts.push(`<b>${((above / S.sim.n) * 100).toFixed(1)}%</b> of pixels ≥ ${S.sim.thr.toFixed(2)} (${fmtArea(above)})`);
  }
  parts.push(`ref ${S.sim.nref} px · ${Math.round(S.sim.ms)} ms`);
  el.innerHTML = parts.join(' · ');
}

function drawRefs() {
  refLayer.clearLayers();
  const half = S.sim.win;
  S.sim.refs.forEach((p, i) => {
    if (half > 0) {
      L.rectangle(windowBounds(p.r - half, p.c - half, 2 * half + 1, 2 * half + 1), {
        color: '#fff', weight: 1.5, fill: false, interactive: false }).addTo(refLayer);
    }
    L.circleMarker(pixelCenter(p.r, p.c), { radius: 6, color: '#ffffff', weight: 2, fillColor: '#ff2d6f', fillOpacity: 1 })
      .bindTooltip(`ref ${i + 1}${S.sim.lock ? ` (${p.year})` : ''}`).addTo(refLayer);
  });
}
function renderRefChips() {
  const el = $('#refs');
  if (!S.sim.refs.length) { el.innerHTML = '<span class="muted">none yet</span>'; return; }
  el.innerHTML = S.sim.refs.map((p, i) =>
    `<span class="chip">${i + 1}: ${p.lat.toFixed(4)}, ${p.lng.toFixed(4)}${S.sim.lock ? ` · ${p.year}` : ''}<button data-i="${i}" title="remove">×</button></span>`).join('');
}
$('#refs').addEventListener('click', (e) => {
  const i = e.target.dataset.i;
  if (i === undefined) return;
  S.sim.refs.splice(+i, 1);
  computeSim();
});

function addRef(latlng, append) {
  const { r, c } = toPixel(latlng);
  if (!inside(r, c)) { status('That point is outside the data', true); return; }
  const ctr = pixelCenter(r, c);
  const ref = { r, c, year: S.year, lat: ctr.lat, lng: ctr.lng };
  S.sim.refs = append ? [...S.sim.refs, ref] : [ref];
  if (S.mode !== 'sim') setMode('sim'); else computeSim();
}

/* ---------------- hover readout & clicks ---------------- */
let hoverLL = null, hoverRaf = 0;
map.on('mousemove', (e) => { hoverLL = e.latlng; if (!hoverRaf) hoverRaf = requestAnimationFrame(readout); });
map.on('mouseout', () => { hoverLL = null; $('#readout').innerHTML = '<span class="muted">Hover over the map</span>'; });
function readout() {
  hoverRaf = 0;
  if (!hoverLL || !META) return;
  const { r, c } = toPixel(hoverLL);
  const sep = '<span class="sep">|</span>';
  let html = `${hoverLL.lat.toFixed(5)}, ${hoverLL.lng.toFixed(5)}`;
  const sh = S.shown;
  if (inside(r, c) && sh) {
    if (sh.kind === 'sim' && S.sim.data) {
      const v = S.sim.data[r * W + c];
      html += sep + (v ? `similarity <b>${dec(v).toFixed(3)}</b>` : 'no data');
    } else if (sh.kind === 'rgb') {
      const rr = r - sh.r0, cc = c - sh.c0;
      if (rr >= 0 && cc >= 0 && rr < sh.h && cc < sh.w) {
        const i = rr * sh.w + cc;
        if (!sh.chans[0][i]) html += sep + 'no data';
        else {
          html += sep + sh.chans.map((a, j) => {
            const q = a[i];
            const val = sh.ranges ? fmt(sh.ranges[j][0] + ((q - 1) / 254) * (sh.ranges[j][1] - sh.ranges[j][0])) : `${Math.round(((q - 1) / 254) * 100)}%`;
            return `<span style="color:var(--${'rgb'[j]})">■</span> ${sh.labels[j]} ${val}`;
          }).join('&nbsp;&nbsp;');
        }
      }
    }
  }
  $('#readout').innerHTML = html;
}

map.on('click', (e) => {
  if (S.mode === 'sim') { addRef(e.latlng, e.originalEvent.shiftKey); return; }
  const { r, c } = toPixel(e.latlng);
  if (!inside(r, c)) return;
  const div = document.createElement('div');
  div.innerHTML = `<div style="margin-bottom:6px">${e.latlng.lat.toFixed(5)}, ${e.latlng.lng.toFixed(5)}</div>`;
  const btn = document.createElement('button');
  btn.className = 'btn small';
  btn.textContent = 'Find similar pixels';
  btn.onclick = () => { map.closePopup(); addRef(e.latlng, false); };
  div.appendChild(btn);
  L.popup({ closeButton: false }).setLatLng(e.latlng).setContent(div).openOn(map);
});

/* ---------------- UI wiring ---------------- */
function seg(id, onPick) {
  $(id).addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    $$(`${id} button`).forEach((x) => x.classList.toggle('active', x === b));
    onPick(b.dataset.v);
  });
}
function setSeg(id, v) { $$(`${id} button`).forEach((x) => x.classList.toggle('active', x.dataset.v === String(v))); }

function setMode(mode) {
  S.mode = mode;
  $$('.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  $('#panel-rgb').hidden = mode !== 'rgb';
  $('#panel-sim').hidden = mode !== 'sim';
  map.getContainer().classList.toggle('picking', mode === 'sim');
  if (mode === 'sim') { refLayer.addTo(map); computeSim(); }
  else { refLayer.remove(); renderRGB(); }
}
$$('.tabs button').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));

seg('#rgb-source', (v) => {
  S.rgb.source = v;
  $('#panel-rgb').classList.toggle('dims-mode', v === 'dims');
  $('#pca-scope-sec').hidden = v === 'dims';
  renderEV(); renderRGB();
});
seg('#pca-scope', async (v) => {
  $('#refit').hidden = v !== 'view';
  if (v === 'view' && (!S.view || S.view.year !== S.year)) {
    if (!(await fitToView())) { setSeg('#pca-scope', 'region'); $('#refit').hidden = true; return; }
  }
  S.rgb.scope = v;
  buildChannelOptions(); renderRGB();
});
$('#refit').addEventListener('click', async () => {
  if (await fitToView()) { S.rgb.scope = 'view'; buildChannelOptions(); renderRGB(); }
});
['r', 'g', 'b'].forEach((ch, i) => {
  $(`#ch-${ch}`).addEventListener('change', (e) => { S.rgb.pcs[i] = +e.target.value; renderEV(); renderRGB(); });
  $(`#dim-${ch}`).addEventListener('change', (e) => {
    S.rgb.dims[i] = clamp(Math.round(+e.target.value || 0), 0, 127);
    e.target.value = S.rgb.dims[i]; renderRGB();
  });
});
$('#shuffle').addEventListener('click', () => {
  const n = S.rgb.source === 'dims' ? 128 : evList().length;
  const pool = [...Array(n).keys()];
  const pick = [0, 1, 2].map(() => pool.splice(Math.floor(Math.random() * pool.length), 1)[0]);
  if (S.rgb.source === 'dims') S.rgb.dims = pick; else S.rgb.pcs = pick;
  buildChannelOptions(); renderRGB();
});
$('#reset-ch').addEventListener('click', () => {
  if (S.rgb.source === 'dims') S.rgb.dims = [0, 1, 2]; else S.rgb.pcs = [0, 1, 2];
  buildChannelOptions(); renderRGB();
});
seg('#stretch', (v) => { S.rgb.stretch = v; $('#pct-controls').hidden = v !== 'pct'; renderRGB(); });
$('#p-lo').addEventListener('change', (e) => { S.rgb.pLo = clamp(+e.target.value, 0, 49); renderRGB(); });
$('#p-hi').addEventListener('change', (e) => { S.rgb.pHi = clamp(+e.target.value, 51, 100); renderRGB(); });

let simRaf = 0;
function scheduleSim() { if (!simRaf) simRaf = requestAnimationFrame(() => { simRaf = 0; renderSim(); }); }
$('#thr').addEventListener('input', (e) => { S.sim.thr = +e.target.value; syncSliders(); scheduleSim(); });
$('#lo').addEventListener('input', (e) => { S.sim.lo = Math.min(+e.target.value, S.sim.hi - 0.005); syncSliders(); scheduleSim(); });
$('#hi').addEventListener('input', (e) => { S.sim.hi = Math.max(+e.target.value, S.sim.lo + 0.005); syncSliders(); scheduleSim(); });
$('#auto-range').addEventListener('click', () => { autoRange(); renderSim(); });
$('#cmap').addEventListener('change', (e) => { S.sim.cmap = e.target.value; renderSim(); });
$('#clear-refs').addEventListener('click', () => { S.sim.refs = []; computeSim(); });
seg('#win', (v) => { S.sim.win = +v; computeSim(); });
$('#lock-year').addEventListener('change', (e) => { S.sim.lock = e.target.checked; computeSim(); });

$('#opacity').addEventListener('input', (e) => {
  overlay.setOpacity(+e.target.value);
  $('#opacity-v').textContent = `${Math.round(e.target.value * 100)}%`;
});
$('#basemap').addEventListener('change', (e) => { basemap.remove(); basemap = basemaps[e.target.value].addTo(map); });
$('#labels').addEventListener('change', (e) => (e.target.checked ? labels.addTo(map) : labels.remove()));
function setShow(on) { $('#show').checked = on; on ? overlay.addTo(map) : overlay.remove(); }
$('#show').addEventListener('change', (e) => setShow(e.target.checked));
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea') || e.metaKey || e.ctrlKey) return;
  if (e.key === 'h' || e.key === 'H') setShow(!$('#show').checked);
});
$('#download').addEventListener('click', () => {
  canvas.toBlob((blob) => {
    const a = document.createElement('a');
    const what = S.mode === 'sim' ? 'similarity' : S.rgb.source === 'dims' ? `dims_${S.rgb.dims.join('-')}` : `pc${S.rgb.pcs.map((k) => k + 1).join('-')}`;
    a.href = URL.createObjectURL(blob);
    a.download = `${META.region}_${S.year}_${what}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
});

$('#year').addEventListener('change', async (e) => {
  S.year = +e.target.value;
  if (S.rgb.scope === 'view' && S.view) {
    const v = S.view;
    if (!(await loadView(v.r0, v.r0 + v.h, v.c0, v.c0 + v.w))) { S.rgb.scope = 'region'; setSeg('#pca-scope', 'region'); }
  }
  buildChannelOptions();
  if (S.mode === 'sim') computeSim(); else renderRGB();
});

/* ---------------- start ---------------- */
(async function init() {
  try {
    META = await (await fetchOk('/api/meta')).json();
  } catch (e) { status(`Could not load data: ${e.message}`, true); return; }
  W = META.width; H = META.height;
  FULL = L.latLngBounds(META.latlon_bounds);
  if (!META.years.length) { status('No years prepared — run explorer/prepare.py', true); return; }
  S.year = META.years[META.years.length - 1];
  $('#region-name').textContent = `${META.name} · ${W} × ${H} px at 10 m`;
  $('#year').innerHTML = META.years.map((y) => `<option>${y}</option>`).join('');
  $('#year').value = S.year;
  $('#cmap').innerHTML = Object.keys(META.colormaps).map((n) => `<option>${n}</option>`).join('');
  $('#cmap').value = S.sim.cmap;
  overlay = new CanvasOverlay(canvas, FULL, { opacity: 0.9, className: 'data-overlay', interactive: false }).addTo(map);
  L.rectangle(FULL, { color: '#ffffff', weight: 1, dashArray: '4 4', fill: false, interactive: false }).addTo(map);
  map.fitBounds(FULL);
  syncSliders();
  buildChannelOptions();
  drawHist();
  renderRGB();
  setInterval(refreshYears, 20000); // pick up years prepared while the page is open
})();

async function refreshYears() {
  try {
    const m = await (await fetchOk('/api/meta')).json();
    if (m.years.join() === META.years.join()) return;
    META.years = m.years; META.pca = m.pca;
    $('#year').innerHTML = META.years.map((y) => `<option>${y}</option>`).join('');
    $('#year').value = S.year;
    status(`Years available: ${META.years.join(', ')}`);
    setTimeout(() => status(null), 4000);
  } catch (_) { /* server busy or restarting */ }
}
