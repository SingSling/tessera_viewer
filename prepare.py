"""Download TESSERA embeddings for a region and precompute PCA.

    python prepare.py                                  # Frankfurt am Main, 2024
    python prepare.py --year 2018 --year 2025          # add more years (2017-2025)
    python prepare.py --region myfarm --name "My farm" --bbox 7.23,50.235,7.37,50.325

--bbox is min_lon,min_lat,max_lon,max_lat in degrees. Once a region exists,
add years with just --region <name> --year <year>.

The embeddings are resampled (nearest neighbour, values are never mixed) from
their UTM grid onto a Web Mercator grid, so they line up exactly with the
OpenStreetMap / satellite basemaps in the browser.

Writes to data/<region>/ (about 1.5 MB per km² per year, e.g. 0.6 GB for 20 x 20 km):
    <year>_emb.npy    int8  (H, W, 128)  raw quantised embeddings, 0 = no data
    <year>_scale.npy  float32 (H, W)     per-pixel scale, 0 = no data
    <year>_norm.npy   float32 (H, W)     L2 norm of the int8 vector (for cosine)
    <year>_pca.npy    uint8 (K, H, W)    PCA components, 1..255, 0 = no data
    <year>_pca.json   explained variance and value ranges of the components
    meta.json         grid and region description
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pyproj import Transformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tessera_io import store, utm_zone, zone_grid

ROOT = Path(__file__).resolve().parent

# Ready-made regions: name, (min_lon, min_lat, max_lon, max_lat)
PRESETS = {
    # ~20 x 20 km: city, Main river, Stadtwald, airport edge, Offenbach, and the
    # arable land north/east of the city (Nieder-Erlenbach, Bergen-Enkheim, Bad Vilbel).
    "frankfurt": ("Frankfurt am Main", (8.54, 50.03, 8.82, 50.21)),
    # ~10 x 10 km arable loess plateau west of Koblenz.
    "maifeld": ("Maifeld, Rhineland-Palatinate", (7.23, 50.235, 7.37, 50.325)),
}
GROUND_RES = 10.0  # metres
N_PCS = 10
STRIP = 256  # rows per read, matches the store's chunk size
MB_PER_KM2 = 1.5
MAX_KM2 = 1600  # 40 x 40 km; larger needs --large


def target_grid(box):
    """Web Mercator grid covering the lon/lat box at ~10 m ground resolution."""
    lon0, lat0, lon1, lat1 = box
    to_merc = Transformer.from_crs(4326, 3857, always_xy=True)
    (x0, x1), (y0, y1) = to_merc.transform([lon0, lon1], [lat0, lat1])
    res = GROUND_RES / np.cos(np.radians((lat0 + lat1) / 2))
    w = int(round((x1 - x0) / res))
    h = int(round((y1 - y0) / res))
    return dict(x0=x0, y0=y1, res=float(res), width=w, height=h)  # y0 = top edge


def source_index(grid, zone):
    """Row/col in the store's UTM zone grid for each Mercator pixel centre."""
    gx0, gy0, gres = zone_grid(zone)
    xs = grid["x0"] + (np.arange(grid["width"]) + 0.5) * grid["res"]
    ys = grid["y0"] - (np.arange(grid["height"]) + 0.5) * grid["res"]
    X, Y = np.meshgrid(xs, ys)
    to_utm = Transformer.from_crs(3857, 32600 + zone, always_xy=True)
    ux, uy = to_utm.transform(X.ravel(), Y.ravel())
    cols = ((np.asarray(ux) - gx0) // gres).astype(np.int64).reshape(X.shape)
    rows = ((gy0 - np.asarray(uy)) // gres).astype(np.int64).reshape(X.shape)
    return rows, cols


def read_source(zone, year, r0, r1, c0, c1):
    """Raw int8 embeddings (h, w, 128) and scales (h, w) for a UTM pixel window."""
    za = store().zone_arrays(zone)
    ti = store().years.index(year)
    h, w = r1 - r0, c1 - c0
    emb = np.zeros((h, w, 128), np.int8)
    starts = list(range(r0 - r0 % STRIP, r1, STRIP))
    done = [0]

    def strip(s):
        a, b = max(s, r0), min(s + STRIP, r1)
        block = np.asarray(za["embeddings"][ti, :, a:b, c0:c1])  # (128, rows, cols)
        emb[a - r0 : b - r0] = np.moveaxis(block, 0, -1)
        done[0] += 1
        print(f"  strip {done[0]}/{len(starts)}", flush=True)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(strip, starts))
    scales = np.asarray(za["scales"][ti, r0:r1, c0:c1]).astype(np.float32)
    return emb, scales


def fit_pca(emb, scale, valid, k=N_PCS, n_sample=100_000):
    """Standardise + PCA on a sample of valid pixels (as `geotessera visualize` does)."""
    idx = np.flatnonzero(valid.ravel())
    rng = np.random.default_rng(0)
    take = rng.choice(idx, size=min(n_sample, len(idx)), replace=False)
    flat_e = emb.reshape(-1, 128)
    flat_s = scale.ravel()
    sample = flat_e[take].astype(np.float32) * flat_s[take, None]
    scaler = StandardScaler().fit(sample)
    pca = PCA(min(k, len(take)), random_state=0).fit(scaler.transform(sample))
    return scaler, pca


def project_all(emb, scale, valid, scaler, pca, chunk=500_000):
    flat_e = emb.reshape(-1, 128)
    flat_s = scale.ravel()
    out = np.full((flat_e.shape[0], pca.n_components_), np.nan, np.float32)
    idx = np.flatnonzero(valid.ravel())
    for i in range(0, len(idx), chunk):
        j = idx[i : i + chunk]
        x = flat_e[j].astype(np.float32) * flat_s[j, None]
        out[j] = pca.transform(scaler.transform(x))
    return out


def quantise(values, valid, lo_p=0.1, hi_p=99.9):
    """float components (N, K) -> uint8 (K, N), 1..255, 0 = no data."""
    v = values[valid]
    lo, hi = np.percentile(v, [lo_p, hi_p], axis=0)
    q = np.zeros((values.shape[1], values.shape[0]), np.uint8)
    scaled = np.clip((v - lo) / (hi - lo), 0, 1) * 254 + 1
    q[:, valid] = np.round(scaled).astype(np.uint8).T
    return q, lo, hi


def prepare_year(region, name, box, year, out):
    grid = target_grid(box)
    zone = utm_zone((box[0] + box[2]) / 2)
    H, W = grid["height"], grid["width"]
    print(f"{name} {year}: {W} x {H} px (UTM zone {zone})", flush=True)

    t = time.time()
    rows, cols = source_index(grid, zone)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    print(f"Reading {r1 - r0} x {c1 - c0} source pixels from the TESSERA store ...", flush=True)
    src, src_sc = read_source(zone, year, r0, r1, c0, c1)
    print(f"  read in {time.time() - t:.0f} s", flush=True)

    emb = src[rows - r0, cols - c0]  # (H, W, 128)
    scale = src_sc[rows - r0, cols - c0]
    del src, src_sc
    valid = np.isfinite(scale) & (scale > 0)
    if not valid.any():
        raise SystemExit(f"No TESSERA data for {year} in this box.")
    emb[~valid] = 0
    scale = np.where(valid, scale, 0).astype(np.float32)
    print(f"  {valid.mean():.1%} of pixels have data", flush=True)

    norm = np.sqrt(np.einsum("ijk,ijk->ij", emb, emb, dtype=np.float32))
    np.save(out / f"{year}_emb.npy", emb)
    np.save(out / f"{year}_scale.npy", scale)
    np.save(out / f"{year}_norm.npy", norm.astype(np.float32))

    print("Fitting PCA ...", flush=True)
    scaler, pca = fit_pca(emb, scale, valid)
    comps = project_all(emb, scale, valid, scaler, pca)
    q, lo, hi = quantise(comps, valid.ravel())
    np.save(out / f"{year}_pca.npy", q.reshape(-1, H, W))
    meta = {"region": region, "name": name, "bbox": list(box), "crs": "EPSG:3857", **grid, "utm_zone": zone}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    # Written last: the server treats a year as ready once this file exists.
    (out / f"{year}_pca.json").write_text(
        json.dumps(
            {
                "explained_variance_ratio": pca.explained_variance_ratio_.round(4).tolist(),
                "lo": lo.tolist(),
                "hi": hi.tolist(),
            }
        )
    )
    print(f"Done {year} in {time.time() - t:.0f} s -> {out}", flush=True)


def parse_bbox(text):
    try:
        box = tuple(float(v) for v in text.split(","))
    except ValueError:
        box = ()
    if len(box) != 4 or not (box[0] < box[2] and box[1] < box[3]):
        raise SystemExit("--bbox must be min_lon,min_lat,max_lon,max_lat (degrees), e.g. 8.54,50.03,8.82,50.21")
    if not (-180 <= box[0] and box[2] <= 180 and -80 <= box[1] and box[3] <= 80):
        raise SystemExit("--bbox must lie within lon -180..180 and lat -80..80")
    return box


def area_km2(box):
    lon0, lat0, lon1, lat1 = box
    return (lon1 - lon0) * 111.32 * np.cos(np.radians((lat0 + lat1) / 2)) * (lat1 - lat0) * 110.57


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="frankfurt", help=f"region folder name; presets: {', '.join(PRESETS)}")
    ap.add_argument("--bbox", help="min_lon,min_lat,max_lon,max_lat for a new region")
    ap.add_argument("--name", help="display name for a new region")
    ap.add_argument("--year", type=int, action="append", help="repeatable; default 2024")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--force", action="store_true", help="re-download years already prepared")
    ap.add_argument("--large", action="store_true", help=f"allow boxes larger than {MAX_KM2} km²")
    args = ap.parse_args()

    out = args.data_dir / args.region
    existing = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else None
    if args.bbox:
        box, name = parse_bbox(args.bbox), args.name or args.region
    elif existing:
        box, name = tuple(existing["bbox"]), existing["name"]
    elif args.region in PRESETS:
        name, box = PRESETS[args.region]
    else:
        raise SystemExit(f"Unknown region '{args.region}'. Give --bbox for a new region, or use one of: {', '.join(PRESETS)}")
    if existing and tuple(existing["bbox"]) != tuple(box):
        raise SystemExit(f"{out} already holds a different box {existing['bbox']}. Use another --region name.")

    km2 = area_km2(box)
    years = args.year or [2024]
    print(f"{name}: {km2:.0f} km², about {km2 * MB_PER_KM2:.0f} MB per year on disk and in server RAM", flush=True)
    if km2 > MAX_KM2 and not args.large:
        raise SystemExit(f"That box is {km2:.0f} km² (limit {MAX_KM2}). Use a smaller box or pass --large.")
    available = store().years
    bad = [y for y in years if y not in available]
    if bad:
        raise SystemExit(f"Years {bad} are not in the store; available: {available}")

    out.mkdir(parents=True, exist_ok=True)
    for year in years:
        if (out / f"{year}_pca.json").exists() and not args.force:
            print(f"{year} already prepared (use --force to redo)")
            continue
        prepare_year(args.region, name, box, year, out)
    print(f"\nStart the explorer with:  python server.py --region {args.region}")


if __name__ == "__main__":
    main()
