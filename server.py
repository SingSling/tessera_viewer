"""Web server for the TESSERA embedding explorer.

    python server.py                          # http://127.0.0.1:8765
    python server.py --region maifeld --port 8766

Data comes from data/<region>/ (run prepare.py first). With one region prepared
it is picked automatically. The browser does the colouring; the server sends
raw uint8/uint16 rasters:

    GET /api/meta                         region grid, years, PCA stats, colormaps
    GET /api/pca?year=&k=                 PCA component k, uint8 (H*W), 0 = no data
    GET /api/dim?year=&d=                 raw embedding dimension d, uint8, 0 = no data
    GET /api/sim?year=&refs=r:c:y,...&win=  cosine similarity to the mean of the
                                          reference pixels, uint16, 0 = no data
    GET /api/viewpca?year=&r0=&r1=&c0=&c1=  PCA refitted on a window, K x h x w uint8
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from pyproj import Transformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
CMAPS = ["viridis", "magma", "inferno", "cividis", "YlGnBu", "gray"]
VIEW_PCS = 6
MAX_REFS = 200  # reference pixels per similarity request
MAX_HALF = 10  # sample window up to 21 x 21 px around each reference
POOL = ThreadPoolExecutor(8)


class Year:
    """One year's embeddings, loaded into RAM (~0.5 GB for Frankfurt)."""

    def __init__(self, d: Path, year: int):
        self.year = year
        self.emb = np.load(d / f"{year}_emb.npy")  # (H, W, 128) int8
        self.scale = np.load(d / f"{year}_scale.npy")
        self.norm = np.load(d / f"{year}_norm.npy")
        self.pca = np.load(d / f"{year}_pca.npy", mmap_mode="r")  # (K, H, W) uint8
        self.valid = self.scale > 0
        self.H, self.W = self.scale.shape
        self.flat = self.emb.reshape(-1, 128)
        self._dims: dict[int, bytes] = {}

    def unit(self, r, c, half):
        """Unit vectors of the valid pixels in a (2*half+1)^2 window."""
        r0, r1 = max(r - half, 0), min(r + half + 1, self.H)
        c0, c1 = max(c - half, 0), min(c + half + 1, self.W)
        e = self.emb[r0:r1, c0:c1].reshape(-1, 128).astype(np.float32)
        n = self.norm[r0:r1, c0:c1].ravel()
        ok = n > 0
        return e[ok] / n[ok, None]

    def similarity(self, q: np.ndarray) -> np.ndarray:
        """Cosine similarity of every pixel to vector q, float32 (H*W), NaN = no data.

        The per-pixel scale cancels in the cosine, so the raw int8 vectors are used.
        """
        q = (q / np.linalg.norm(q)).astype(np.float32)
        n = self.flat.shape[0]
        out = np.empty(n, np.float32)
        norm = self.norm.ravel()
        step = 1 << 17

        def chunk(i):
            j = slice(i, min(i + step, n))
            out[j] = (self.flat[j].astype(np.float32) @ q) / np.where(norm[j] > 0, norm[j], np.nan)

        list(POOL.map(chunk, range(0, n, step)))
        return out

    def dim(self, d: int) -> bytes:
        if d not in self._dims:
            v = self.emb[..., d].astype(np.float32) * self.scale
            lo, hi = np.percentile(v[self.valid], [0.5, 99.5])
            q = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1) * 254 + 1
            q = np.where(self.valid, np.round(q), 0).astype(np.uint8)
            self._dims[d] = q.tobytes()
        return self._dims[d]

    def view_pca(self, r0, r1, c0, c1, k=VIEW_PCS):
        e = self.emb[r0:r1, c0:c1].reshape(-1, 128)
        s = self.scale[r0:r1, c0:c1].ravel()
        ok = s > 0
        idx = np.flatnonzero(ok)
        if len(idx) < 50:
            raise ValueError("too few pixels with data in this view")
        rng = np.random.default_rng(0)
        take = rng.choice(idx, size=min(60_000, len(idx)), replace=False)
        sample = e[take].astype(np.float32) * s[take, None]
        scaler = StandardScaler().fit(sample)
        k = min(k, 128, len(take))
        pca = PCA(k, random_state=0).fit(scaler.transform(sample))
        comps = np.zeros((k, e.shape[0]), np.float32)
        for i in range(0, len(idx), 400_000):
            j = idx[i : i + 400_000]
            comps[:, j] = pca.transform(scaler.transform(e[j].astype(np.float32) * s[j, None])).T
        lo = np.percentile(comps[:, ok], 0.1, axis=1)
        hi = np.percentile(comps[:, ok], 99.9, axis=1)
        q = np.clip((comps - lo[:, None]) / (hi - lo + 1e-9)[:, None], 0, 1) * 254 + 1
        q = np.where(ok[None, :], np.round(q), 0).astype(np.uint8)
        info = {
            "k": k, "h": r1 - r0, "w": c1 - c0, "r0": r0, "c0": c0,
            "explained_variance_ratio": pca.explained_variance_ratio_.round(4).tolist(),
            "lo": lo.tolist(), "hi": hi.tolist(),
        }
        return q.tobytes(), info


class Region:
    def __init__(self, directory: Path, max_years: int = 2):
        self.dir = directory
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.max_years = max_years
        self._years: OrderedDict[int, Year] = OrderedDict()
        self._lock = threading.Lock()

    def years(self):
        # *_pca.json is the last file prepare.py writes, so its presence means the year is complete.
        return sorted(int(p.name.split("_")[0]) for p in self.dir.glob("*_pca.json"))

    def get(self, year: int) -> Year:
        with self._lock:
            if year not in self._years:
                if year not in self.years():
                    raise ValueError(f"year {year} not prepared")
                self._years[year] = Year(self.dir, year)
                while len(self._years) > self.max_years:  # keep RAM bounded
                    self._years.popitem(last=False)
            self._years.move_to_end(year)
            return self._years[year]

    def meta_json(self):
        m = dict(self.meta)
        to_ll = Transformer.from_crs(3857, 4326, always_xy=True)
        x1 = m["x0"] + m["width"] * m["res"]
        y1 = m["y0"] - m["height"] * m["res"]
        (lon0, lon1), (lat0, lat1) = to_ll.transform([m["x0"], x1], [y1, m["y0"]])
        m["latlon_bounds"] = [[lat0, lon0], [lat1, lon1]]
        m["years"] = self.years()
        m["pca"] = {y: json.loads((self.dir / f"{y}_pca.json").read_text()) for y in m["years"]}
        m["colormaps"] = colormaps()
        return m


@lru_cache(1)
def colormaps():
    import matplotlib

    out = {}
    for name in CMAPS:
        cm = matplotlib.colormaps[name].resampled(256)
        out[name] = (cm(np.arange(256))[:, :3] * 255).round().astype(int).ravel().tolist()
    return out


def encode_sim(s: np.ndarray) -> bytes:
    """float similarity in [-1, 1] -> uint16, 1..65535; 0 = no data."""
    q = np.where(np.isfinite(s), np.round((np.clip(s, -1, 1) + 1) / 2 * 65534) + 1, 0)
    return q.astype("<u2").tobytes()


def make_handler(region: Region):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if "/api/" in self.path:
                print(f"{self.command} {self.path[:120]}", flush=True)

        def send(self, body: bytes, ctype: str, headers: dict | None = None, status=200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj, status=200):
            self.send(json.dumps(obj).encode(), "application/json", status=status)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path.startswith("/api/"):
                    return self.api(u.path[5:], q)
                return self.static(u.path)
            except (ValueError, KeyError, IndexError) as e:
                return self.send_json({"error": str(e)}, 400)
            except Exception as e:  # keep the server alive; report to the page
                return self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

        def static(self, path):
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            f = (STATIC / name).resolve()
            if STATIC not in f.parents or not f.is_file():
                return self.send(b"not found", "text/plain", status=404)
            ctype = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}.get(
                f.suffix, "application/octet-stream"
            )
            self.send(f.read_bytes(), ctype + "; charset=utf-8")

        def api(self, name, q):
            if name == "meta":
                return self.send_json(region.meta_json())
            y = region.get(int(q["year"]))
            if name == "pca":
                return self.send(np.ascontiguousarray(y.pca[int(q["k"])]).tobytes(), "application/octet-stream")
            if name == "dim":
                d = int(q["d"])
                if not 0 <= d < 128:
                    raise ValueError("dimension must be 0..127")
                return self.send(y.dim(d), "application/octet-stream")
            if name == "sim":
                half = min(max(int(q.get("win", 0)), 0), MAX_HALF)
                refs = q["refs"].split(",")
                if len(refs) > MAX_REFS:
                    raise ValueError(f"at most {MAX_REFS} reference pixels")
                vecs = []
                for ref in refs:
                    r, c, ry = (int(v) for v in ref.split(":"))
                    vecs.append(region.get(ry).unit(r, c, half))
                vecs = np.vstack(vecs)
                if not len(vecs):
                    raise ValueError("reference pixels have no data")
                s = y.similarity(vecs.mean(0))
                return self.send(encode_sim(s), "application/octet-stream", {"X-Ref-Pixels": str(len(vecs))})
            if name == "viewpca":
                r0, r1 = max(int(q["r0"]), 0), min(int(q["r1"]), y.H)
                c0, c1 = max(int(q["c0"]), 0), min(int(q["c1"]), y.W)
                if r1 - r0 < 8 or c1 - c0 < 8:
                    raise ValueError("view is outside the data")
                body, info = y.view_pca(r0, r1, c0, c1)
                return self.send(body, "application/octet-stream", {"X-Info": json.dumps(info)})
            return self.send_json({"error": f"unknown endpoint {name}"}, 404)

    return Handler


def main():
    ap = argparse.ArgumentParser(description="TESSERA embedding explorer")
    ap.add_argument("--region", help="folder in data/ (default: the only/first prepared region)")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other machines (no password!)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-years", type=int, default=2, help="years kept in RAM at once")
    args = ap.parse_args()

    prepared = sorted(p.parent.name for p in args.data_dir.glob("*/meta.json"))
    if not prepared:
        raise SystemExit(f"No data in {args.data_dir}. Download some first:  python prepare.py")
    name = args.region or prepared[0]
    if name not in prepared:
        raise SystemExit(f"Region '{name}' not prepared. Available: {', '.join(prepared)}")
    region = Region(args.data_dir / name, args.max_years)
    years = region.years()
    if not years:
        raise SystemExit(f"No complete year in {region.dir}. Run:  python prepare.py --region {name}")
    print(f"{region.meta['name']}: years {years}" + (f"  (other regions: {', '.join(p for p in prepared if p != name)})" if len(prepared) > 1 else ""), flush=True)
    region.get(years[-1])  # warm up the latest year
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(region))
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    print(f"Explorer running at http://{shown}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
