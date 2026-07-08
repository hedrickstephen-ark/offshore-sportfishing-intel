"""
Offshore Sportfishing Intel — backend service.
NOAA satellite intel for US East Coast offshore fishing.

Region-parameterized: every raster/vector endpoint accepts a bounding box
(s, w, n, e) so the same service powers any stretch of coast — Stuart FL,
Ocean City MD, and anywhere in between. If no bbox is passed it defaults to
the Stuart, FL box.

Serves the derived layers the Leaflet front end overlays on the satellite base:
  - SST + chlorophyll rasters (rendered here so they always draw, gap-tolerant)
  - Bait / edge composite  (SST fronts + chlorophyll edges + SSH eddies)
  - Sea-surface-height anomaly  (warm/cold-core eddies)  [live nesdisSSH1day]
  - Animated surface currents  (leaflet-velocity format)  [live nesdisSSH1day]
  - Point depth  (ETOPO1)

Every source is free/public (NOAA CoastWatch ERDDAP). When a live fetch is
briefly unreachable the endpoint falls back to a realistic synthetic field so
the map always renders — /sources reports which is in use.

Run:
  pip install -r requirements.txt
  uvicorn app:app --reload --port 8000
Verify offline:
  python app.py            # writes selftest PNGs from synthetic data, no network
"""

import io
import os
import time
import math
import datetime as dt
import warnings
from collections import OrderedDict

import numpy as np

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import requests

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".matplotlib-cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import Normalize
from PIL import Image
from fastapi import FastAPI, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ----------------------------------------------------------------------------
# Default region (Stuart, FL) — used when no bbox is supplied.
# bbox order everywhere is (south, west, north, east).
# ----------------------------------------------------------------------------
DEF_BBOX = (26.6, -80.3, 27.9, -79.2)

ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
SST_DS = "jplMURSST41"      # analysed_sst (Kelvin), 0.01 deg, daily
CHL_DS = "erdMH1chla1day"   # chlorophyll (mg m^-3), ~0.041 deg, daily
BATHY_DS = "etopo180"       # altitude (m, neg = below sea), global
SSH_DS = "nesdisSSH1day"    # sla (m) + ugos/vgos (m/s), NRT altimetry, global

HTTP_TIMEOUT = (5, 20)       # connect, read
MISS_TTL = 600               # retry failed/sparse NOAA pulls after 10 minutes
PNG_TTL = 1800
JSON_TTL = 900

HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Offshore-Sportfishing-Intel/0.1"})

_ALT_CACHE = {}   # bbox-key -> {"t": epoch, "data": {...} or None}
_LAST_SOURCE = {}

app = FastAPI(title="Offshore Sportfishing Intel — backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def bbox_from(s, w, n, e):
    """Build a (s,w,n,e) tuple from query params, falling back to default."""
    if None in (s, w, n, e):
        return DEF_BBOX
    s, n = sorted((float(s), float(n)))
    w, e = sorted((float(w), float(e)))
    return (s, w, n, e)


# ----------------------------------------------------------------------------
# ERDDAP grid fetch -> (lats asc, lons asc, 2D grid) ; None on failure
# ----------------------------------------------------------------------------
def _erddap_url(ds, var, date, bbox, stride=1):
    s, w, n, e = bbox
    t = f"({date}T00:00:00Z)"
    lat = f"({s}):{stride}:({n})"
    lon = f"({w}):{stride}:({e})"
    return f"{ERDDAP}/{ds}.json?{var}[{t}][{lat}][{lon}]"


def fetch_grid(ds, var, date, bbox, stride=1, timeout=HTTP_TIMEOUT):
    try:
        r = HTTP.get(_erddap_url(ds, var, date, bbox, stride), timeout=timeout)
        r.raise_for_status()
        t = r.json()["table"]; cols = t["columnNames"]; rows = t["rows"]
        ila, ilo, iv = cols.index("latitude"), cols.index("longitude"), cols.index(var)
        lats = np.array(sorted({row[ila] for row in rows}), float)
        lons = np.array(sorted({row[ilo] for row in rows}), float)
        li = {v: k for k, v in enumerate(lats)}
        lj = {v: k for k, v in enumerate(lons)}
        g = np.full((lats.size, lons.size), np.nan)
        for row in rows:
            if row[iv] is not None:
                g[li[row[ila]], lj[row[ilo]]] = row[iv]
        if np.isfinite(g).sum() < g.size * 0.1:      # mostly clouds -> miss
            return None
        return lats, lons, g
    except Exception:
        return None


IMG_HEADERS = {"Cache-Control": f"public, max-age={PNG_TTL}"}   # let browsers cache tiles

_GRID_CACHE = {}   # (ds,var,date,bbox,stride) -> (epoch, result/None); avoids re-fetching
_PNG_CACHE = OrderedDict()
_JSON_CACHE = OrderedDict()


def _remember(cache, key, value, max_items=64):
    cache[key] = (time.time(), value)
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


def _cached(cache, key, ttl):
    hit = cache.get(key)
    if not hit:
        return None
    t, value = hit
    if time.time() - t >= ttl:
        cache.pop(key, None)
        return None
    cache.move_to_end(key)
    return value


def _bbox_key(bbox, places=3):
    return tuple(round(x, places) for x in bbox)


def _norm01(arr):
    a = np.asarray(arr, float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a)
    lo, hi = np.nanmin(a), np.nanmax(a)
    out = (a - lo) / (hi - lo + 1e-9)
    out[~finite] = np.nan
    return out


def cached_fetch_grid(ds, var, date, bbox, stride, ttl=1800):
    key = (ds, var, date, _bbox_key(bbox), stride)
    now = time.time()
    c = _GRID_CACHE.get(key)
    if c and now - c[0] < (ttl if c[1] is not None else MISS_TTL):
        return c[1]
    g = fetch_grid(ds, var, date, bbox, stride)
    _GRID_CACHE[key] = (now, g)
    return g


def fetch_altimetry(bbox, timeout=HTTP_TIMEOUT):
    """sla + ugos + vgos for the region at the LATEST time. dict or None."""
    s, w, n, e = bbox
    box = f"[(last)][({s}):({n})][({w}):({e})]"
    q = f"?sla{box},ugos{box},vgos{box}"
    try:
        r = HTTP.get(f"{ERDDAP}/{SSH_DS}.json{q}", timeout=timeout)
        r.raise_for_status()
        t = r.json()["table"]; cols = t["columnNames"]; rows = t["rows"]
        ila, ilo = cols.index("latitude"), cols.index("longitude")
        isla, iu, iv = cols.index("sla"), cols.index("ugos"), cols.index("vgos")
        lats = np.array(sorted({row[ila] for row in rows}), float)
        lons = np.array(sorted({row[ilo] for row in rows}), float)
        li = {v: k for k, v in enumerate(lats)}
        lj = {v: k for k, v in enumerate(lons)}
        sh = (lats.size, lons.size)
        sla = np.full(sh, np.nan); ug = np.full(sh, np.nan); vg = np.full(sh, np.nan)
        for row in rows:
            a, b = li[row[ila]], lj[row[ilo]]
            if row[isla] is not None: sla[a, b] = row[isla]
            if row[iu] is not None:   ug[a, b] = row[iu]
            if row[iv] is not None:   vg[a, b] = row[iv]
        if np.isfinite(sla).sum() < 4:
            return None
        return {"lats": lats, "lons": lons, "sla": sla, "ugos": ug, "vgos": vg}
    except Exception:
        return None


def get_altimetry(bbox):
    """Cached altimetry per region (3 h TTL). None if unreachable."""
    key = _bbox_key(bbox, 2)
    now = time.time()
    c = _ALT_CACHE.get(key)
    if c and now - c["t"] < (3 * 3600 if c["data"] is not None else MISS_TTL):
        return c["data"]
    d = fetch_altimetry(bbox)
    _ALT_CACHE[key] = {"t": now, "data": d}
    return d


# ----------------------------------------------------------------------------
# Generic synthetic fallback fields (an offshore, meandering warm front).
# Works for any bbox: warm water offshore (east), cooler inshore/north.
# ----------------------------------------------------------------------------
def _axes(bbox, n=101, m=101):
    s, w, nn, e = bbox
    return np.linspace(s, nn, n), np.linspace(w, e, m)


def _front_lon(lats, bbox, phase=0.0):
    s, w, nn, e = bbox
    span_lat = (nn - s) or 1e-6
    frac = 0.55 + 0.10 * np.sin((lats - s) / span_lat * 10 + phase)
    return w + (e - w) * frac


def synth_sst(bbox, phase=0.0):
    lats, lons = _axes(bbox)
    LO, LA = np.meshgrid(lons, lats)
    edge = _front_lon(lats, bbox, phase)[:, None]
    warm = 1 / (1 + np.exp(-(LO - edge) / max((bbox[3] - bbox[1]) * 0.02, 1e-4)))
    span_lat = (bbox[2] - bbox[0]) or 1e-6
    north_cool = 1.4 * ((LA - bbox[0]) / span_lat)
    sst = 299.0 + 3.5 * warm - north_cool + 0.15 * np.random.randn(*LO.shape)
    return lats, lons, sst


def synth_chl(bbox, phase=0.0):
    lats, lons = _axes(bbox)
    LO, LA = np.meshgrid(lons, lats)
    edge = _front_lon(lats, bbox, phase)[:, None]
    green = 1 / (1 + np.exp((LO - edge) / max((bbox[3] - bbox[1]) * 0.02, 1e-4)))
    coastal = np.exp(-(((LO - bbox[1]) / max((bbox[3] - bbox[1]) * 0.15, 1e-4)) ** 2))
    chl = 0.05 + 0.7 * green + 0.2 * coastal + 0.03 * np.random.randn(*LO.shape)
    return lats, lons, np.clip(chl, 0.02, None)


def synth_ssh(bbox, phase=0.0):
    lats, lons = _axes(bbox)
    LO, LA = np.meshgrid(lons, lats)
    s, w, nn, e = bbox
    cy1, cx1 = s + 0.6 * (nn - s), w + 0.6 * (e - w)
    cy2, cx2 = s + 0.35 * (nn - s), w + 0.45 * (e - w)
    r2 = max(((nn - s) * (e - w)) * 0.02, 1e-4)
    ssh = (0.25 * np.exp(-(((LO - cx1) ** 2 + (LA - cy1) ** 2) / r2))
           - 0.18 * np.exp(-(((LO - cx2) ** 2 + (LA - cy2) ** 2) / r2)))
    return lats, lons, ssh


def _synth_currents(bbox, phase=0.0):
    lats, lons = _axes(bbox, 61, 61)
    LO, LA = np.meshgrid(lons, lats)
    edge = _front_lon(lats, bbox, phase)[:, None]
    width = max((bbox[3] - bbox[1]) * 0.06, 1e-4)
    core = np.exp(-(((LO - edge) / width) ** 2))
    v = 1.8 * core - 0.12
    u = 0.6 * core * np.cos((LA - bbox[0]) * 3.2 + phase)
    return lats, lons, u, v


def get_field(kind, date, bbox, phase=0.0):
    """Return (lats, lons, grid, source). Live ERDDAP if reachable, else synth."""
    if kind == "sst":
        got = cached_fetch_grid(SST_DS, "analysed_sst", date, bbox, stride=3)
        if got:
            _LAST_SOURCE["sst"] = f"live: {SST_DS}"
            return (*got, _LAST_SOURCE["sst"])
        _LAST_SOURCE["sst"] = "demo (synthetic)"
        return (*synth_sst(bbox, phase), _LAST_SOURCE["sst"])
    if kind == "chl":
        got = cached_fetch_grid(CHL_DS, "chlorophyll", date, bbox, stride=1)
        if got:
            _LAST_SOURCE["chl"] = f"live: {CHL_DS}"
            return (*got, _LAST_SOURCE["chl"])
        _LAST_SOURCE["chl"] = "demo (synthetic)"
        return (*synth_chl(bbox, phase), _LAST_SOURCE["chl"])
    if kind == "ssh":
        alt = get_altimetry(bbox)
        if alt is not None:
            _LAST_SOURCE["ssh"] = f"live: {SSH_DS}"
            return alt["lats"], alt["lons"], alt["sla"], _LAST_SOURCE["ssh"]
        _LAST_SOURCE["ssh"] = "demo (synthetic)"
        return (*synth_ssh(bbox, phase), _LAST_SOURCE["ssh"])
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# Front / edge math
# ----------------------------------------------------------------------------
def front_strength(grid):
    g = grid.copy()
    mask = np.isnan(g)
    if mask.any():
        g[mask] = np.nanmean(g)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    finite = mag[np.isfinite(mag)]
    if finite.size == 0:
        return np.zeros_like(mag)
    lo, hi = np.nanpercentile(finite, 5), np.nanpercentile(finite, 95)
    out = np.clip((mag - lo) / (hi - lo + 1e-9), 0, 1)
    out[mask] = 0.0
    if mask.any():                      # kill false "edges" along the coastline
        ring = np.zeros_like(mask)
        ring[:-1, :] |= mask[1:, :]; ring[1:, :] |= mask[:-1, :]
        ring[:, :-1] |= mask[:, 1:]; ring[:, 1:] |= mask[:, :-1]
        out[ring & ~mask] = 0.0
    return out


def resample_to(rlat, rlon, lats, lons, grid):
    li = np.abs(rlat[:, None] - lats[None, :]).argmin(axis=1)
    lj = np.abs(rlon[:, None] - lons[None, :]).argmin(axis=1)
    return grid[np.ix_(li, lj)]


def build_composite(date, bbox, w_sst=0.55, w_chl=0.30, w_ssh=0.15):
    lat_s, lon_s, sst, _ = get_field("sst", date, bbox)
    lat_c, lon_c, chl, _ = get_field("chl", date, bbox)
    lat_h, lon_h, ssh, _ = get_field("ssh", date, bbox)

    sf = front_strength(sst)
    cf = resample_to(lat_s, lon_s, lat_c, lon_c,
                     front_strength(np.log10(np.clip(chl, 1e-3, None))))
    hf = resample_to(lat_s, lon_s, lat_h, lon_h,
                     np.clip(np.abs(ssh) / (np.nanmax(np.abs(ssh)) + 1e-9), 0, 1))
    comp = w_sst * sf + w_chl * cf + w_ssh * hf
    comp = np.clip(comp / max(comp.max(), 1e-9), 0, 1)
    comp[np.isnan(sst)] = np.nan          # transparent over land / cloud
    return comp, lat_s, lon_s


# ----------------------------------------------------------------------------
# Rendering -> PNG (north-up, transparent where weak)
# ----------------------------------------------------------------------------
def array_to_png(arr, lats=None, cmap="magma", vmin=0.0, vmax=1.0,
                 alpha_mode="value", alpha_gamma=1.3, upscale=6):
    """alpha_mode='mask' -> opaque over data, transparent over land/cloud (SST, chl).
       alpha_mode='value' -> fade weak values (composite edges)."""
    a = np.array(arr, float)
    if lats is not None and lats[0] < lats[-1]:
        a = a[::-1, :]
    finite = np.isfinite(a)
    rgba = matplotlib.colormaps[cmap](Normalize(vmin, vmax)(np.nan_to_num(a)))
    if alpha_mode == "mask":
        rgba[..., 3] = np.where(finite, 0.92, 0.0)
    else:
        rgba[..., 3] = np.where(finite, np.clip(np.power(np.nan_to_num(a), alpha_gamma), 0, 1), 0.0)
    img = Image.fromarray((rgba * 255).astype(np.uint8))
    img = img.resize((img.width * upscale, img.height * upscale), Image.BICUBIC)
    buf = io.BytesIO(); img.save(buf, "PNG")
    return buf.getvalue()


def _today_minus(days):
    return (dt.date.today() - dt.timedelta(days=days)).isoformat()


def _q(s, w, n, e):
    return bbox_from(s, w, n, e)


def _png_response(key, build):
    body = _cached(_PNG_CACHE, key, PNG_TTL)
    if body is None:
        body = build()
        _remember(_PNG_CACHE, key, body)
    return Response(body, media_type="image/png", headers=IMG_HEADERS)


def _json_cached(key, build):
    data = _cached(_JSON_CACHE, key, JSON_TTL)
    if data is None:
        data = build()
        _remember(_JSON_CACHE, key, data)
    return data


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
# Serve the dashboard from the same origin as the API, so one Render web service
# hosts everything (visit the service URL and the map loads + calls its own API).
_HERE = os.path.dirname(__file__)
_HTML_CANDIDATES = [
    os.path.join(_HERE, "offshore-sportfishing-intel.html"),        # flat deploy layout
    os.path.join(_HERE, "..", "offshore-sportfishing-intel.html"),  # local repo layout
]


def _html_path():
    for p in _HTML_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


@app.get("/", include_in_schema=False)
def index():
    p = _html_path()
    if p:
        return FileResponse(p, media_type="text/html")
    return Response("Dashboard file not found; API is running.", media_type="text/plain")


@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}


@app.get("/meta")
def meta():
    return {
        "default_bbox_swne": DEF_BBOX,
        "default_date": _today_minus(2),
        "sources": {"sst": SST_DS, "chl": CHL_DS, "ssh_currents": SSH_DS,
                    "bathymetry": BATHY_DS, "provider": "NOAA CoastWatch ERDDAP"},
    }


@app.get("/sources")
def sources():
    return {"last_used": _LAST_SOURCE,
            "note": "live = fetched from NOAA ERDDAP; demo = synthetic fallback"}


@app.get("/composite.png")
def composite_png(date: str = Query(None),
                  s: float = None, w: float = None, n: float = None, e: float = None,
                  w_sst: float = 0.55, w_chl: float = 0.30, w_ssh: float = 0.15):
    date = date or _today_minus(2)
    bbox = _q(s, w, n, e)
    key = ("composite", date, _bbox_key(bbox), round(w_sst, 3), round(w_chl, 3), round(w_ssh, 3))
    def build():
        comp, lats, _ = build_composite(date, bbox, w_sst, w_chl, w_ssh)
        return array_to_png(comp, lats, cmap="magma")
    return _png_response(key, build)


@app.get("/sst.png")
def sst_png(date: str = Query(None),
            s: float = None, w: float = None, n: float = None, e: float = None):
    date = date or _today_minus(2)
    bbox = _q(s, w, n, e)
    key = ("sst", date, _bbox_key(bbox))
    def build():
        lats, lons, sst, _ = get_field("sst", date, bbox)
        return array_to_png(_norm01(sst - 273.15), lats, cmap="turbo", alpha_mode="mask")
    return _png_response(key, build)


@app.get("/chl.png")
def chl_png(date: str = Query(None),
            s: float = None, w: float = None, n: float = None, e: float = None):
    date = date or _today_minus(2)
    bbox = _q(s, w, n, e)
    key = ("chl", date, _bbox_key(bbox))
    def build():
        lats, lons, chl, _ = get_field("chl", date, bbox)
        lg = np.log10(np.clip(chl, 1e-3, None))
        lg[~np.isfinite(np.asarray(chl, float))] = np.nan
        return array_to_png(_norm01(lg), lats, cmap="YlGn", alpha_mode="mask")
    return _png_response(key, build)


@app.get("/ssh.png")
def ssh_png(date: str = Query(None),
            s: float = None, w: float = None, n: float = None, e: float = None):
    date = date or _today_minus(2)
    bbox = _q(s, w, n, e)
    key = ("ssh", date, _bbox_key(bbox))
    def build():
        lats, lons, ssh, _ = get_field("ssh", date, bbox)
        finite_abs = np.abs(ssh[np.isfinite(ssh)])
        lim = max(float(finite_abs.max()) if finite_abs.size else 0.0, 1e-6)
        norm = np.nan_to_num((ssh + lim) / (2 * lim), nan=0.5)
        a = np.nan_to_num(np.abs(ssh) / lim, nan=0.0)
        rgba = matplotlib.colormaps["RdBu_r"](Normalize(0, 1)(norm))
        if lats[0] < lats[-1]:
            rgba = rgba[::-1, :]; a = a[::-1, :]
        rgba[..., 3] = np.clip(a, 0, 0.85)
        img = Image.fromarray((rgba * 255).astype(np.uint8))
        img = img.resize((img.width * 4, img.height * 4), Image.BILINEAR)
        buf = io.BytesIO(); img.save(buf, "PNG")
        return buf.getvalue()
    return _png_response(key, build)


@app.get("/currents.json")
def currents_json(date: str = Query(None),
                  s: float = None, w: float = None, n: float = None, e: float = None):
    """leaflet-velocity format. Live NOAA geostrophic currents when reachable."""
    date = date or _today_minus(2)
    bbox = _q(s, w, n, e)
    key = ("currents", date, _bbox_key(bbox))
    def build():
        alt = get_altimetry(bbox)
        if alt is not None:
            lats, lons, u, v = alt["lats"], alt["lons"], alt["ugos"], alt["vgos"]
            _LAST_SOURCE["currents"] = f"live: {SSH_DS}"
        else:
            lats, lons, u, v = _synth_currents(bbox, time.time() % (2 * math.pi))
            _LAST_SOURCE["currents"] = "demo (synthetic)"
        u = np.nan_to_num(u); v = np.nan_to_num(v)
        ny, nx = u.shape
        hdr = {
            "nx": nx, "ny": ny,
            "lo1": float(lons[0]), "la1": float(lats[-1]),
            "lo2": float(lons[-1]), "la2": float(lats[0]),
            "dx": float(abs(lons[1] - lons[0])), "dy": float(abs(lats[1] - lats[0])),
            "parameterCategory": 2, "parameterNumberName": "current",
            "refTime": date + "T00:00:00Z", "parameterUnit": "m.s-1",
        }
        uf = u[::-1, :].ravel().round(3).tolist()
        vf = v[::-1, :].ravel().round(3).tolist()
        return [
            {"header": {**hdr, "parameterNumber": 2}, "data": uf},
            {"header": {**hdr, "parameterNumber": 3}, "data": vf},
        ]
    return _json_cached(key, build)


@app.get("/depth")
def depth(lat: float = Query(...), lon: float = Query(...)):
    """Charted depth at a point from NOAA ETOPO1 bathymetry (etopo180)."""
    url = f"{ERDDAP}/{BATHY_DS}.json?altitude[({lat})][({lon})]"
    try:
        r = HTTP.get(url, timeout=HTTP_TIMEOUT); r.raise_for_status()
        alt_m = r.json()["table"]["rows"][0][-1]
        if alt_m is None:
            raise ValueError("no value")
        alt_m = float(alt_m); depth_m = -alt_m
        return {
            "lat": lat, "lon": lon,
            "elevation_m": round(alt_m, 1),
            "depth_m": round(depth_m, 1),
            "depth_ft": round(depth_m * 3.28084, 1),
            "depth_fathoms": round(depth_m / 1.8288, 1),
            "is_water": depth_m > 0,
            "source": "NOAA ETOPO1", "live": True,
        }
    except Exception:
        return {"lat": lat, "lon": lon, "live": False, "source": "unavailable"}


if __name__ == "__main__":
    # Offline self-test: prove every renderer works with no network.
    for name, png in [
        ("selftest_composite.png", composite_png("2000-01-01").body),
        ("selftest_sst.png", sst_png("2000-01-01").body),
        ("selftest_chl.png", chl_png("2000-01-01").body),
        ("selftest_ssh.png", ssh_png("2000-01-01").body),
    ]:
        open(name, "wb").write(png)
        print(f"OK {name} ({len(png)} bytes)")
    for reg, bb in [("FL", (26.6, -80.3, 27.9, -79.2)), ("MD", (37.2, -75.1, 38.6, -73.3))]:
        cur = currents_json("2000-01-01", *(bb[0], bb[1], bb[2], bb[3]))
        print(f"OK currents.json [{reg}] {len(cur[0]['data'])} cells")
