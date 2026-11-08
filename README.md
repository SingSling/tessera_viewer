# TESSERA Embedding Explorer

An interactive map for exploring [TESSERA](https://arxiv.org/abs/2506.20380)
satellite embeddings (128 numbers per 10 m pixel, one set per year,
summarising what the land surface did through that year).

- **RGB composite.** Colour the map with any three PCA components or raw
  embedding dimensions. You can refit the PCA on just the area in view to get
  much more contrast between fields.
- **Similarity search.** Click a pixel, or shift-click several, and every other
  pixel is coloured by how similar it is. A threshold shows where (and how many
  hectares) look like your reference, even in another year.

The repository contains **no data**. A script downloads the embeddings for
your chosen area directly from the public TESSERA store (anonymous, no account
needed).

## Quick start

Needs **Python 3.12 or newer**. The default area, Frankfurt am Main
(20 × 20 km), needs about 0.6 GB of disk and RAM per year.

```bash
git clone https://github.com/SingSling/tessera-explorer.git
cd tessera-explorer
```

**With [uv](https://docs.astral.sh/uv/)** (recommended; it fetches Python 3.12
for you if your system Python is older):

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python prepare.py           # downloads Frankfurt, 2024 (~3 min)
uv run python server.py            # open http://127.0.0.1:8765
```

**With pip** (check `python3 --version` shows 3.12 or newer first):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python prepare.py
python server.py
```

The other examples below use `python ...`. With uv, write `uv run python ...`.

### More years

```bash
python prepare.py --year 2018 --year 2025          # 2017–2025 are available
```

New years show up in an open page within about 20 seconds.

### Your own area

```bash
python prepare.py --region myfarm --name "My farm" --bbox 7.23,50.235,7.37,50.325
python server.py --region myfarm
```

`--bbox` is `min_lon,min_lat,max_lon,max_lat` in degrees (for example, copy the
corners from OpenStreetMap). Disk and server RAM are about 1.5 MB per km² per
year. Boxes over 1,600 km² need `--large`. A 10–25 km square works well.
Once a region exists, add years with `python prepare.py --region myfarm --year 2020`.
Presets: `frankfurt` (default) and `maifeld` (arable plateau west of Koblenz).

### Server options

| Option | Default | |
|---|---|---|
| `--region` | the only (or first) prepared region | folder name in `data/` |
| `--port` | 8765 | |
| `--host` | 127.0.0.1 | `0.0.0.0` makes it reachable from other machines on your network. There is no password, so only do this on a trusted network. |
| `--max-years` | 2 | years kept in RAM at once |
| `--data-dir` | `./data` | |

## Using the map

**RGB composite**
- *PCA components*: choose three of the first 10 components as R, G, B. The bars
  show how much variance each one explains.
- *PCA fitted on → Current view* refits the PCA on the pixels you can see.
  Fitted on the whole region, the first components mostly separate city,
  forest and fields. Fitted on a farmland view, they separate individual fields and crops.
- *Raw dimensions*: any three of the 128 embedding dimensions.
- *Colour stretch*: histogram equalisation (default), percentile clip, or ±2.5σ.
- Clicking the map gives a popup with **Find similar pixels**.

**Similarity**
- **Click** a pixel to colour all pixels by cosine similarity to it.
  **Shift+click** adds more pixels, and their average is used.
- *Sample around each click*: use a 3×3 to 11×11 block instead of one pixel
  (less noise, good for "this field").
- *Hide below*: pixels below the threshold turn transparent. The line under the
  histogram gives the share and area above it. Clicking the histogram also sets the threshold.
- *Keep reference from the year it was picked*: switch year to search for the same
  signature in another year (for example, "where did this 2024 crop grow in 2018?").

**Display**: overlay opacity, satellite / OpenStreetMap / light basemap, place
labels, **H** toggles the overlay, and *Download overlay PNG* saves the current layer
at full resolution.

## How it works

- `prepare.py` reads the int8 embeddings for the box from the TESSERA v1.1
  Icechunk store (`s3://tessera-embeddings/v1.1/dclimate.icechunk`, via
  [geotessera](https://github.com/ucam-eo/geotessera)). It then resamples them
  nearest-neighbour from UTM onto a Web Mercator grid so pixels line up with web
  basemaps, and fits a PCA (standardise + PCA on 100k sampled pixels, like
  `geotessera visualize`).
- `server.py` is a standard-library HTTP server. It keeps the selected year in
  RAM and sends raw rasters. Cosine similarity over 4 million pixels takes about 0.1 s.
  The per-pixel quantisation scale cancels out in the cosine, so it works on the int8 vectors directly.
- `static/app.js` (Leaflet) does all colouring in the browser, so the sliders react instantly.

```
prepare.py      download + resample + PCA  ->  data/<region>/
server.py       HTTP API + static files
tessera_io.py   minimal reader for the TESSERA store
static/         index.html, app.js, style.css
```

## Licence, data and credits

- Code: [MIT](LICENSE). The licence covers this repository's code only, not the
  embeddings you download with it.

- Embeddings: **TESSERA**. *TESSERA: Temporal Embeddings of Surface Spectra
  for Earth Representation and Analysis*,
  [arXiv:2506.20380](https://arxiv.org/abs/2506.20380). Check the current
  licence and acceptable-use terms on the TESSERA / geotessera pages before
  sharing maps made from them. The licence has been listed differently in
  different places (CC-BY-SA on the AWS Open Data registry, CC0 on newer model
  cards). Credit TESSERA either way.
- Basemaps: © OpenStreetMap contributors; imagery and labels © Esri and partners.
  The Esri tiles are used without an API key, which is fine for personal use.
  For a public deployment, check Esri's terms.

## Troubleshooting

- **`Address already in use`**: another copy is running. Stop it or use `--port 8766`.
- **Download is slow or stalls**: the store is on public S3, so a VPN or proxy can
  slow it down. Re-running `prepare.py` is safe, because finished years are skipped.
- **Blank map tiles**: the page needs internet access for the basemaps and Leaflet.
