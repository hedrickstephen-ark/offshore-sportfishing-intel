"""
Offshore Sportfishing Intel — backend service.
NOAA satellite ocean intelligence for US East Coast offshore fishing.

Viewport-driven: every raster/vector endpoint accepts the live map bounding box
(s, w, n, e), so overlays cover whatever water the angler is looking at — no
fixed fishing-area box. Large zoom-outs are span-clamped and auto-strided so the
service stays fast.

Serves the layers the Leaflet front end overlays on the satellite base:
  - SST + chlorophyll rasters (rendered here so they always draw, gap-tolerant)
  - Bait / edge composite  (SST fronts + chlorophyll edges + SSH eddies)
  - Sea-surface-height anomaly  (warm/cold-core eddies)      [nesdisSSH1day]
  - Geostrophic surface currents  (leaflet-velocity)          [nesdisSSH1day]
  - Wind-driven (Ekman) currents  (leaflet-velocity)          [erdQCwindproducts7day]
  - Ocean surface wind  (leaflet-velocity)                    [erdQCwindproducts7day]
  - Ekman upwelling raster                                    [erdQCwindproducts7day]
  - Point depth  (ETOPO1)

Data sources (all free/public, no key):
  SST        jplMURSST41                    (MUR, 1 km gap-free L4)          [pfeg]
  Chlorophyll noaacwNPPN20S3ASCIDINEOF2kmDaily (VIIRS+OLCI gap-filled 2 km) [coastwatch]
  SSH/eddies + geostrophic currents  nesdisSSH1day (NRT altimetry)          [pfeg]
  Wind / Ekman currents / upwelling  erdQCwindproducts7day (Metop-C ASCAT)  [pfeg]
  Bathymetry  etopo180 (ETOPO1)                                             [pfeg]

If a live fetch is momentarily unreachable the endpoint returns a clearly
labeled modeled field so the map still renders; /sources reports live-vs-modeled
per layer and modeled fields are never presented as observed data.

Run:
  pip install -r requirements.txt
  uvicorn app:app --reload --port 8000
Verify offline:
  python app.py            # writes selftest PNGs from modeled data, no network
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
# Map extent handling. bbox order everywhere is (south, west, north, east).
# The dashboard always sends the live map viewport; when no bbox is supplied we
# fall back to a broad US East Coast box so a bare request still renders.
# ----------------------------------------------------------------------------
DEF_BBOX = (26.6, -80.3, 27.9, -79.2)
MAX_SPAN_DEG = 40.0          # clamp absurdly large viewports (whole-ocean zoom-outs)
TARGET_CELLS = 220           # per axis: auto-stride keeps fetched grids ~this size

# NOAA CoastWatch ERDDAP nodes (both free, no key required)
ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"     # West Coast node
ERDDAP_CW = "https://coastwatch.noaa.gov/erddap/griddap"       # primary CoastWatch node

SST_DS = "jplMURSST41"       # analysed_sst (°C on this mirror), 0.01°, daily, gap-free L4 (MUR)
CHL_DS = "noaacwNPPN20S3ASCIDINEOF2kmDaily"   # chlor_a, VIIRS+OLCI gap-filled DINEOF 2km
CHL_VAR = "chlor_a"
BATHY_DS = "etopo180"        # altitude (m, neg = below sea), global
SSH_DS = "nesdisSSH1day"     # sla (m) + ugos/vgos (m/s), NRT altimetry, global
# Metop-C ASCAT merged product: wind vectors, wind-driven (Ekman) currents,
# and Ekman upwelling in one dataset (0.25°, NRT). Same variable schema across
# composite windows — try freshest first, fall through to a gap-filled window
# when a short composite is too sparse over the requested box.
WINDPROD_CHAIN = ["erdQCwindproducts1day", "erdQCwindproducts3day", "erdQCwindproducts7day"]
WIND_MIN_FRAC = 0.20         # need at least this fraction of finite cells to accept

# Honest label shown when a live NOAA pull is momentarily unavailable and the
# service renders a modeled field instead. Never presented as observed data.
MODELED = "modeled estimate (live data unavailable)"

HTTP_TIMEOUT = (5, 20)       # connect, read
MISS_TTL = 600               # retry failed/sparse NOAA pulls after 10 minutes
PNG_TTL = 1800
JSON_TTL = 900

HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Offshore-Sportfishing-Intel/1.0"})

_ALT_CACHE = {}   # bbox-key -> {"t": epoch, "data": {...} or None}
_LAST_SOURCE = {}

app = FastAPI(title="Offshore Sportfishing Intel — backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def clamp_bbox(bbox):
    """Clamp an over-large viewport to MAX_SPAN_DEG around its center."""
    s, w, n, e = bbox
    cs, cw = (s + n) / 2, (w + e) / 2
    if n - s > MAX_SPAN_DEG:
        s, n = cs - MAX_SPAN_DEG / 2, cs + MAX_SPAN_DEG / 2
    if e - w > MAX_SPAN_DEG:
        w, e = cw - MAX_SPAN_DEG / 2, cw + MAX_SPAN_DEG / 2
    return (max(-89.0, s), w, min(89.0, n), e)


def bbox_from(s, w, n, e):
    """Build a clamped (s,w,n,e) tuple from query params, falling back to default."""
    if None in (s, w, n, e):
        return DEF_BBOX
    s, n = sorted((float(s), float(n)))
    w, e = sorted((float(w), float(e)))
    return clamp_bbox((s, w, n, e))


def auto_stride(bbox, res_deg):
    """Stride that keeps a fetched grid near TARGET_CELLS per axis for this bbox."""
    s, w, n, e = bbox
    span = max(n - s, e - w)
    return max(1, int(span / max(res_deg, 1e-6) / TARGET_CELLS))


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


# ----------------------------------------------------------------------------
# General ERDDAP fetchers that handle an optional altitude dimension, a `last`
# time selector, and datasets whose latitude axis descends (VIIRS chl, the
# ASCAT wind products). Used for the newer layers; MUR SST keeps fetch_grid.
# ----------------------------------------------------------------------------
def _coord_str(bbox, stride, when, alt, lat_desc):
    s, w, n, e = bbox
    t = "(last)" if when == "last" else f"({when}T00:00:00Z)"
    la = f"({n}):{stride}:({s})" if lat_desc else f"({s}):{stride}:({n})"
    lo = f"({w}):{stride}:({e})"
    return f"[{t}]" + (f"[({alt})]" if alt is not None else "") + f"[{la}][{lo}]"


def fetch_scalar(base, ds, var, bbox, stride, when="last", alt=None,
                 lat_desc=False, fill=None):
    """One scalar variable -> (lats asc, lons asc, 2D grid) or None."""
    try:
        url = f"{base}/{ds}.json?{var}{_coord_str(bbox, stride, when, alt, lat_desc)}"
        r = HTTP.get(url, timeout=HTTP_TIMEOUT); r.raise_for_status()
        t = r.json()["table"]; cols = t["columnNames"]; rows = t["rows"]
        ila, ilo, iv = cols.index("latitude"), cols.index("longitude"), cols.index(var)
        lats = np.array(sorted({row[ila] for row in rows}), float)
        lons = np.array(sorted({row[ilo] for row in rows}), float)
        li = {v: k for k, v in enumerate(lats)}; lj = {v: k for k, v in enumerate(lons)}
        g = np.full((lats.size, lons.size), np.nan)
        for row in rows:
            val = row[iv]
            if val is not None and (fill is None or val != fill):
                g[li[row[ila]], lj[row[ilo]]] = val
        if np.isfinite(g).sum() < 4:
            return None
        return lats, lons, g
    except Exception:
        return None


def fetch_vector2(base, ds, uvar, vvar, bbox, stride, when="last", alt=None,
                  lat_desc=False, fill=None):
    """Two vector-component variables in one request -> dict or None."""
    try:
        box = _coord_str(bbox, stride, when, alt, lat_desc)
        q = f"?{uvar}{box},{vvar}{box}"
        r = HTTP.get(f"{base}/{ds}.json{q}", timeout=HTTP_TIMEOUT); r.raise_for_status()
        t = r.json()["table"]; cols = t["columnNames"]; rows = t["rows"]
        ila, ilo = cols.index("latitude"), cols.index("longitude")
        iu, iv = cols.index(uvar), cols.index(vvar)
        lats = np.array(sorted({row[ila] for row in rows}), float)
        lons = np.array(sorted({row[ilo] for row in rows}), float)
        li = {v: k for k, v in enumerate(lats)}; lj = {v: k for k, v in enumerate(lons)}
        sh = (lats.size, lons.size)
        U = np.full(sh, np.nan); V = np.full(sh, np.nan)
        for row in rows:
            a, b = li[row[ila]], lj[row[ilo]]
            if row[iu] is not None and row[iu] != fill: U[a, b] = row[iu]
            if row[iv] is not None and row[iv] != fill: V[a, b] = row[iv]
        if np.isfinite(U).sum() < 4:
            return None
        return {"lats": lats, "lons": lons, "u": U, "v": V}
    except Exception:
        return None


_GEN_CACHE = {}   # name+bbox+stride+when -> (epoch, result/None)


def cached_scalar(name, ttl=1800, **kw):
    key = (name, _bbox_key(kw["bbox"]), kw["stride"], kw.get("when"))
    now = time.time(); c = _GEN_CACHE.get(key)
    if c and now - c[0] < (ttl if c[1] is not None else MISS_TTL):
        return c[1]
    g = fetch_scalar(**kw)
    _GEN_CACHE[key] = (now, g)
    return g


def cached_vector(name, ttl=3600, **kw):
    key = (name, _bbox_key(kw["bbox"]), kw["stride"], kw.get("when"))
    now = time.time(); c = _GEN_CACHE.get(key)
    if c and now - c[0] < (ttl if c[1] is not None else MISS_TTL):
        return c[1]
    d = fetch_vector2(**kw)
    _GEN_CACHE[key] = (now, d)
    return d


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
    sst = 26.5 + 3.5 * warm - north_cool + 0.15 * np.random.randn(*LO.shape)  # °C
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
    """Return (lats, lons, grid, source). Live ERDDAP if reachable, else modeled."""
    if kind == "sst":
        st = max(3, auto_stride(bbox, 0.01))          # MUR is 0.01°; cap the pull
        got = cached_fetch_grid(SST_DS, "analysed_sst", date, bbox, stride=st)
        if got:
            lats, lons, g = got
            if np.isfinite(g).any() and np.nanmedian(g) > 100:
                g = g - 273.15                        # some MUR mirrors serve Kelvin
            _LAST_SOURCE["sst"] = f"live: {SST_DS}"
            return lats, lons, g, _LAST_SOURCE["sst"]
        _LAST_SOURCE["sst"] = MODELED
        return (*synth_sst(bbox, phase), MODELED)
    if kind == "chl":
        st = max(1, auto_stride(bbox, 0.0208))         # VIIRS DINEOF is ~0.021°
        # science-quality gap-filled product lags a few days -> take newest available
        got = cached_scalar("chl", base=ERDDAP_CW, ds=CHL_DS, var=CHL_VAR, bbox=bbox,
                            stride=st, when="last", alt=0.0, lat_desc=True, fill=-999.0)
        if got:
            _LAST_SOURCE["chl"] = f"live: {CHL_DS}"
            return (*got, _LAST_SOURCE["chl"])
        _LAST_SOURCE["chl"] = MODELED
        return (*synth_chl(bbox, phase), MODELED)
    if kind == "ssh":
        alt = get_altimetry(bbox)
        if alt is not None:
            _LAST_SOURCE["ssh"] = f"live: {SSH_DS}"
            return alt["lats"], alt["lons"], alt["sla"], _LAST_SOURCE["ssh"]
        _LAST_SOURCE["ssh"] = MODELED
        return (*synth_ssh(bbox, phase), MODELED)
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# Wind / Ekman-current / upwelling layers (Metop-C ASCAT, erdQCwindproducts7day).
# Each returns live data when reachable, else a clearly labeled modeled field.
# ----------------------------------------------------------------------------
def _synth_wind(bbox, phase=0.0):
    lats, lons = _axes(bbox, 41, 41)
    LO, LA = np.meshgrid(lons, lats)
    u = 6.0 + 2.0 * np.sin((LA - bbox[0]) * 2.0 + phase)
    v = 3.0 * np.cos((LO - bbox[1]) * 2.0 + phase)
    return lats, lons, u, v


def _synth_ekman(bbox, phase=0.0):
    lats, lons, u, v = _synth_currents(bbox, phase)
    return lats, lons, 0.03 * u, 0.03 * v


def _synth_upwelling(bbox, phase=0.0):
    lats, lons, ssh = synth_ssh(bbox, phase)
    return lats, lons, ssh * 2e-5


def _cover_ok(grid):
    a = np.asarray(grid, float)
    return a.size and np.isfinite(a).sum() >= max(8, WIND_MIN_FRAC * a.size)


def get_wind(bbox):
    st = max(1, auto_stride(bbox, 0.333))
    for ds in WINDPROD_CHAIN:
        d = cached_vector("wind:" + ds, base=ERDDAP, ds=ds, uvar="wind_u", vvar="wind_v",
                          bbox=bbox, stride=st, when="last", alt=10.0, lat_desc=True, fill=-9999.0)
        if d is not None and _cover_ok(d["u"]):
            _LAST_SOURCE["wind"] = f"live: {ds}"
            return d, _LAST_SOURCE["wind"]
    lats, lons, u, v = _synth_wind(bbox, time.time() % (2 * math.pi))
    _LAST_SOURCE["wind"] = MODELED
    return {"lats": lats, "lons": lons, "u": u, "v": v}, MODELED


def get_ekman(bbox):
    st = max(1, auto_stride(bbox, 0.333))
    for ds in WINDPROD_CHAIN:
        d = cached_vector("ekman:" + ds, base=ERDDAP, ds=ds, uvar="ekman_current_u",
                          vvar="ekman_current_v", bbox=bbox, stride=st, when="last",
                          alt=10.0, lat_desc=True, fill=-99999.0)
        if d is not None and _cover_ok(d["u"]):
            _LAST_SOURCE["ekman"] = f"live: {ds}"
            return d, _LAST_SOURCE["ekman"]
    lats, lons, u, v = _synth_ekman(bbox, time.time() % (2 * math.pi))
    _LAST_SOURCE["ekman"] = MODELED
    return {"lats": lats, "lons": lons, "u": u, "v": v}, MODELED


def get_upwelling(bbox):
    st = max(1, auto_stride(bbox, 0.333))
    for ds in WINDPROD_CHAIN:
        got = cached_scalar("upw:" + ds, base=ERDDAP, ds=ds, var="ekman_upwelling",
                            bbox=bbox, stride=st, when="last", alt=10.0, lat_desc=True,
                            fill=-99999.0)
        if got and _cover_ok(got[2]):
            _LAST_SOURCE["upwelling"] = f"live: {ds}"
            return (*got, _LAST_SOURCE["upwelling"])
    _LAST_SOURCE["upwelling"] = MODELED
    return (*_synth_upwelling(bbox, 0.3), MODELED)


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
# Conditions report — a rule-based "why it's fishing here today" read.
# Turns the same NOAA grids into absolute metrics (a real °F/mi break, a real
# eddy, a real current in knots) and a species-aware plain-English verdict.
# Deterministic, offline-capable, no LLM. build_report() -> dict.
# ----------------------------------------------------------------------------
def _spacing_km(lats, lons):
    dlat = abs(np.mean(np.diff(lats))) if lats.size > 1 else 0.02
    dlon = abs(np.mean(np.diff(lons))) if lons.size > 1 else 0.02
    ky = 111.0 * dlat
    kx = 111.0 * math.cos(math.radians(float(np.mean(lats)))) * dlon
    return max(ky, 1e-6), max(kx, 1e-6)


def _bearing(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon))
    brg = (math.degrees(math.atan2(y, x)) + 360) % 360
    return ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW",
            "WSW", "W", "WNW", "NW", "NNW"][int((brg + 11.25) // 22.5) % 16]


def _nm(lat1, lon1, lat2, lon2):
    R = 3440.065  # nautical miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.asin(min(1, math.sqrt(a)))


def _clip100(x):
    return float(max(0.0, min(100.0, x)))


def build_report(date, bbox, ref=None):
    """Return a structured conditions read + plain-English verdict for a box."""
    lat_s, lon_s, sst, sst_src = get_field("sst", date, bbox)
    _, _, chl, _ = get_field("chl", date, bbox)
    lat_h, lon_h, sla, _ = get_field("ssh", date, bbox)
    alt = get_altimetry(bbox)
    up_lat, up_lon, up, _ = get_upwelling(bbox)
    if ref is None:
        ref = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    finite = np.isfinite(sst)
    sstF = sst * 9 / 5 + 32          # get_field returns SST in °C
    ky, kx = _spacing_km(lat_s, lon_s)
    gy, gx = np.gradient(np.nan_to_num(sstF, nan=float(np.nanmean(sstF))))
    grad = np.hypot(gy / ky, gx / kx)          # °F per km
    grad[~finite] = np.nan
    # suppress the coastline ring so land edges don't read as a break
    if (~finite).any():
        m = ~finite; ring = np.zeros_like(m)
        ring[:-1, :] |= m[1:, :]; ring[1:, :] |= m[:-1, :]
        ring[:, :-1] |= m[:, 1:]; ring[:, 1:] |= m[:, :-1]
        grad[ring & finite] = np.nan
    gf = grad[np.isfinite(grad)]
    break_permile = float(np.nanpercentile(gf, 99)) * 1.609 if gf.size else 0.0
    warm_side = float(np.nanpercentile(sstF[finite], 90)) if finite.any() else 0.0
    # location of the strongest break
    bk_lat = bk_lon = None
    if gf.size:
        idx = np.unravel_index(np.nanargmax(grad), grad.shape)
        bk_lat, bk_lon = float(lat_s[idx[0]]), float(lon_s[idx[1]])

    # chlorophyll color line
    chl_ok = np.isfinite(np.asarray(chl, float))
    lg = np.log10(np.clip(np.asarray(chl, float), 1e-3, None))
    lg[~chl_ok] = np.nan
    cy, cx = np.gradient(np.nan_to_num(lg, nan=float(np.nanmean(lg[chl_ok])) if chl_ok.any() else 0.0))
    cmag = np.hypot(cy, cx); cmag[~chl_ok] = np.nan
    chl_edge_raw = float(np.nanpercentile(cmag[np.isfinite(cmag)], 97)) if chl_ok.any() else 0.0
    clean_chl = float(np.nanpercentile(np.asarray(chl, float)[chl_ok], 10)) if chl_ok.any() else float("nan")

    # eddies (SSH anomaly)
    sla_abs = np.abs(sla[np.isfinite(sla)]) if np.isfinite(sla).any() else np.array([])
    eddy_amp = float(sla_abs.max()) if sla_abs.size else 0.0
    eddy_sign = 0.0
    if np.isfinite(sla).any():
        j = np.unravel_index(np.nanargmax(np.abs(sla)), sla.shape)
        eddy_sign = float(sla[j])

    # surface current (geostrophic, knots)
    if alt is not None:
        spd = np.hypot(np.nan_to_num(alt["ugos"]), np.nan_to_num(alt["vgos"])) * 1.94384
        cur_mean, cur_max = float(np.nanmean(spd)), float(np.nanmax(spd))
    else:
        cur_mean = cur_max = 0.0

    up_abs = np.abs(up[np.isfinite(up)]) if np.isfinite(up).any() else np.array([])
    up_amp = float(np.nanpercentile(up_abs, 95)) if up_abs.size else 0.0

    # ---- scores (0-100, absolute thresholds tuned for offshore breaks) ----
    s_front = _clip100(break_permile / 2.0 * 100)     # ~2°F/mi = pegged
    s_chl = _clip100(chl_edge_raw / 0.30 * 100)
    s_eddy = _clip100(eddy_amp / 0.30 * 100)          # 0.30 m = strong eddy
    s_cur = _clip100(cur_mean / 1.5 * 100)            # 1.5 kt mean = strong
    s_up = _clip100(up_amp / 1.5e-5 * 100)
    scores = {"sst_front": round(s_front), "chl_edge": round(s_chl),
              "eddy": round(s_eddy), "current": round(s_cur), "upwelling": round(s_up)}
    index = round(0.35 * s_front + 0.20 * s_chl + 0.20 * s_eddy
                  + 0.15 * s_cur + 0.10 * s_up)
    grade = ("Strong" if index >= 70 else "Moderate" if index >= 50
             else "Light" if index >= 30 else "Flat")
    strong = bool(index >= 68 or s_front >= 78 or s_eddy >= 82)

    # ---- factual, science-based read: describes the physical oceanography only,
    #      no species or catch implications ----
    bullets = []
    if s_front >= 35 and bk_lat is not None:
        dist = _nm(ref[0], ref[1], bk_lat, bk_lon)
        brg = _bearing(ref[0], ref[1], bk_lat, bk_lon)
        bullets.append(
            f"Sea-surface temperature break ~{break_permile:.1f}°F/mile; warm side "
            f"~{warm_side:.0f}°F, about {dist:.0f} nm {brg}.")
    elif finite.any():
        bullets.append(f"No sharp thermal break in view; warmest water ~{warm_side:.0f}°F, "
                       f"gradients weak across the area.")
    if s_chl >= 35 and np.isfinite(clean_chl):
        bullets.append(
            f"Chlorophyll color edge present — clearer water (~{clean_chl:.2f} mg/m³) meeting "
            f"higher-chlorophyll water.")
    if s_eddy >= 40:
        kind = ("Warm-core (clockwise, high SSH)" if eddy_sign > 0
                else "Cold-core (counter-clockwise, low SSH)")
        bullets.append(f"{kind} eddy in range (sea-surface height {eddy_sign:+.2f} m).")
    if s_cur >= 30:
        bullets.append(f"Surface current ~{cur_mean:.1f} kt (peak {cur_max:.1f} kt).")
    if s_up >= 45:
        bullets.append("Ekman upwelling active nearby (wind-driven vertical transport).")
    if not bullets:
        bullets.append("Uniform water — weak fronts and light current across the view.")

    headline = {
        "Strong": "Strong frontal / eddy structure in view.",
        "Moderate": "Moderate structure — some breaks and current present.",
        "Light": "Light structure — mostly uniform water.",
        "Flat": "Flat — no significant structure in view.",
    }[grade]

    modeled = MODELED in sst_src
    return {
        "date": date, "bbox_swne": bbox, "grade": grade, "index": index,
        "strong": strong, "headline": headline, "bullets": bullets,
        "scores": scores,
        "metrics": {
            "break_degF_per_mile": round(break_permile, 2),
            "warm_side_degF": round(warm_side, 1),
            "clean_chl_mg_m3": round(clean_chl, 3) if np.isfinite(clean_chl) else None,
            "eddy_ssh_m": round(eddy_sign, 3),
            "current_mean_kt": round(cur_mean, 2),
            "current_peak_kt": round(cur_max, 2),
        },
        "data": "modeled" if modeled else "live",
    }


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
_CODE_DIR = os.path.join(os.path.dirname(__file__), "..")
_THIS_DIR = os.path.dirname(__file__)
# prefer the sanitized/minified build (for hosting); fall back to the dev file.
# Same-dir candidates support the flat deploy layout (GitHub repo root).
DASHBOARD_CANDIDATES = [
    os.path.normpath(os.path.join(_CODE_DIR, "dist", "offshore-sportfishing-intel.min.html")),
    os.path.normpath(os.path.join(_CODE_DIR, "offshore-sportfishing-intel.html")),
    os.path.join(_THIS_DIR, "offshore-sportfishing-intel.min.html"),
    os.path.join(_THIS_DIR, "offshore-sportfishing-intel.html"),
]


@app.get("/")
def dashboard():
    """Serve the dashboard same-origin with the API (no file:// / CORS quirks).
    Serves the minified build when present, else the readable dev file."""
    for p in DASHBOARD_CANDIDATES:
        if os.path.exists(p):
            return FileResponse(p, media_type="text/html")
    return Response("dashboard HTML not found next to the backend", status_code=404)


@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}


@app.get("/meta")
def meta():
    return {
        "default_bbox_swne": DEF_BBOX,
        "default_date": _today_minus(2),
        "viewport_driven": True,
        "sources": {
            "sst": SST_DS,
            "chl": CHL_DS,
            "ssh_eddies": SSH_DS,
            "geostrophic_currents": SSH_DS,
            "wind": "erdQCwindproducts[1/3/7]day (freshest with coverage)",
            "ekman_currents": "erdQCwindproducts[1/3/7]day (freshest with coverage)",
            "upwelling": "erdQCwindproducts[1/3/7]day (freshest with coverage)",
            "bathymetry": BATHY_DS,
            "provider": "NOAA CoastWatch ERDDAP",
        },
    }


@app.get("/sources")
def sources():
    return {"last_used": _LAST_SOURCE,
            "note": "live = fetched from NOAA ERDDAP; "
                    "'modeled estimate' = fallback field rendered when a live pull "
                    "is momentarily unavailable (never presented as observed data)"}


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
        lats, lons, sst, _ = get_field("sst", date, bbox)   # already °C
        return array_to_png(_norm01(sst), lats, cmap="turbo", alpha_mode="mask")
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


def velocity_payload(lats, lons, u, v, date, unit="m.s-1", name="current"):
    """Pack a u/v field into the two-header leaflet-velocity JSON format."""
    u = np.nan_to_num(u); v = np.nan_to_num(v)
    ny, nx = u.shape
    hdr = {
        "nx": nx, "ny": ny,
        "lo1": float(lons[0]), "la1": float(lats[-1]),
        "lo2": float(lons[-1]), "la2": float(lats[0]),
        "dx": float(abs(lons[1] - lons[0])), "dy": float(abs(lats[1] - lats[0])),
        "parameterCategory": 2, "parameterNumberName": name,
        "refTime": date + "T00:00:00Z", "parameterUnit": unit,
    }
    uf = u[::-1, :].ravel().round(3).tolist()
    vf = v[::-1, :].ravel().round(3).tolist()
    return [
        {"header": {**hdr, "parameterNumber": 2}, "data": uf},
        {"header": {**hdr, "parameterNumber": 3}, "data": vf},
    ]


@app.get("/currents.json")
def currents_json(date: str = Query(None),
                  s: float = None, w: float = None, n: float = None, e: float = None):
    """Geostrophic surface currents (leaflet-velocity). Live NOAA altimetry when reachable."""
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
            _LAST_SOURCE["currents"] = MODELED
        return velocity_payload(lats, lons, u, v, date)
    return _json_cached(key, build)


@app.get("/ekman.json")
def ekman_json(date: str = Query(None),
               s: float = None, w: float = None, n: float = None, e: float = None):
    """Wind-driven (Ekman) surface currents — where the wind pushes the top layer of water."""
    date = date or _today_minus(2)
    bbox = _q(s, w, n, e)
    key = ("ekman", date, _bbox_key(bbox))
    def build():
        d, _ = get_ekman(bbox)
        return velocity_payload(d["lats"], d["lons"], d["u"], d["v"], date, name="ekman")
    return _json_cached(key, build)


@app.get("/wind.json")
def wind_json(date: str = Query(None),
              s: float = None, w: float = None, n: float = None, e: float = None):
    """Ocean surface wind (leaflet-velocity), Metop-C ASCAT."""
    date = date or _today_minus(2)
    bbox = _q(s, w, n, e)
    key = ("wind", date, _bbox_key(bbox))
    def build():
        d, _ = get_wind(bbox)
        return velocity_payload(d["lats"], d["lons"], d["u"], d["v"], date, name="wind")
    return _json_cached(key, build)


@app.get("/upwelling.png")
def upwelling_png(date: str = Query(None),
                  s: float = None, w: float = None, n: float = None, e: float = None):
    """Ekman upwelling raster. Warm = upwelling (nutrient-rich water rising)."""
    date = date or _today_minus(2)
    bbox = _q(s, w, n, e)
    key = ("upwelling", date, _bbox_key(bbox))
    def build():
        lats, lons, up, _ = get_upwelling(bbox)
        finite = np.abs(up[np.isfinite(up)])
        lim = max(float(np.nanpercentile(finite, 95)) if finite.size else 0.0, 1e-9)
        norm = np.nan_to_num((up + lim) / (2 * lim), nan=0.5)
        a = np.nan_to_num(np.clip(np.abs(up) / lim, 0, 1), nan=0.0)
        arr = np.array(norm, float)
        alpha = np.array(a, float)
        if lats[0] < lats[-1]:
            arr = arr[::-1, :]; alpha = alpha[::-1, :]
        rgba = matplotlib.colormaps["RdYlBu_r"](Normalize(0, 1)(arr))
        rgba[..., 3] = np.clip(alpha, 0, 0.85)
        img = Image.fromarray((rgba * 255).astype(np.uint8))
        img = img.resize((img.width * 8, img.height * 8), Image.BILINEAR)
        buf = io.BytesIO(); img.save(buf, "PNG")
        return buf.getvalue()
    return _png_response(key, build)


@app.get("/report")
def report(date: str = Query(None),
           s: float = None, w: float = None, n: float = None, e: float = None,
           ref_lat: float = None, ref_lon: float = None):
    """Rule-based 'why it's fishing here today' read for the given box."""
    date = date or _today_minus(2)
    bbox = _q(s, w, n, e)
    ref = (ref_lat, ref_lon) if ref_lat is not None and ref_lon is not None else None
    key = ("report", date, _bbox_key(bbox), ref)
    return _json_cached(key, lambda: build_report(date, bbox, ref))


# ----------------------------------------------------------------------------
# Named fishing grounds, served from the backend (kept out of the client source).
# cat drives the map legend/colour. approx=True coordinates are ballpark — meant
# for orientation, not navigation; refine against charts / your own waypoints.
# Extend freely; a sourced pass (Marlin Magazine, NOAA ENC, local knowledge) can
# 10x this list with verified numbers.
# ----------------------------------------------------------------------------
SPOT_CATS = {
    "canyon":  {"label": "Canyon",            "color": "#ff5d5d"},
    "lump":    {"label": "Lump / shoal",      "color": "#ff8c1a"},
    "reef":    {"label": "Reef / wreck",      "color": "#34d0c0"},
    "ledge":   {"label": "Ledge / structure", "color": "#ffd000"},
    "edge":    {"label": "Edge / hotspot",    "color": "#ff2d95"},
    "inlet":   {"label": "Inlet / ramp",      "color": "#00e0ff"},
}

SPOTS = {
    "stuart": [
        {"n": "St. Lucie Inlet", "lat": 27.165, "lon": -80.155, "cat": "inlet", "t": "Main run-out for the Stuart fleet"},
        {"n": "Fort Pierce Inlet", "lat": 27.470, "lon": -80.290, "cat": "inlet", "t": "North run-out", "approx": True},
        {"n": "~120 ft Ledge", "lat": 27.19, "lon": -80.10, "cat": "ledge", "t": "Structure ~5 mi off Stuart; funnels bait"},
        {"n": "Peck's Lake", "lat": 27.12, "lon": -80.14, "cat": "ledge", "t": "Nearshore reef/structure", "approx": True},
        {"n": "Six Mile Reef", "lat": 27.10, "lon": -80.05, "cat": "reef", "t": "Reef line SE of the inlet", "approx": True},
        {"n": "Eight Mile Reef", "lat": 27.15, "lon": -80.00, "cat": "reef", "t": "Reef/hardbottom", "approx": True},
        {"n": "Loran Tower area", "lat": 27.20, "lon": -79.98, "cat": "ledge", "t": "Structure/wreck zone", "approx": True},
        {"n": "Bethel Shoal", "lat": 27.45, "lon": -80.12, "cat": "lump", "t": "Shoal + tower to the north", "approx": True},
        {"n": "Capron Shoal", "lat": 27.52, "lon": -80.20, "cat": "lump", "t": "Shoal off Ft. Pierce", "approx": True},
        {"n": "Push Button Hill", "lat": 27.18, "lon": -79.92, "cat": "edge", "t": "Popular sailfish ground", "approx": True},
        {"n": "Sailfish Alley (edge)", "lat": 27.22, "lon": -79.85, "cat": "edge", "t": "Troll the temp/color break", "approx": True},
        {"n": "Gulf Stream edge", "lat": 27.25, "lon": -79.55, "cat": "edge", "t": "Warm blue water; watch SST + chl break"},
        {"n": "The 27 Fathom edge", "lat": 27.20, "lon": -79.62, "cat": "edge", "t": "Blue-water dropoff", "approx": True},
    ],
    "oceancity": [
        {"n": "Ocean City Inlet", "lat": 38.317, "lon": -75.083, "cat": "inlet", "t": "Main run-out"},
        {"n": "Jackspot", "lat": 38.0867, "lon": -74.7517, "cat": "lump", "t": "Shoal ~20 mi out; early tuna, blues"},
        {"n": "The Hot Dog", "lat": 38.20, "lon": -74.50, "cat": "lump", "t": "Lump on the 30-fathom line", "approx": True},
        {"n": "The Fingers", "lat": 38.30, "lon": -74.32, "cat": "lump", "t": "Bottom relief off the shelf", "approx": True},
        {"n": "Great Eastern Reef", "lat": 38.10, "lon": -74.85, "cat": "reef", "t": "Artificial reef", "approx": True},
        {"n": "Bass Grounds", "lat": 38.32, "lon": -74.90, "cat": "reef", "t": "Hardbottom / wreck area", "approx": True},
        {"n": "Norfolk Canyon", "lat": 37.05, "lon": -74.75, "cat": "canyon", "t": "Southern canyon", "approx": True},
        {"n": "Washington Canyon", "lat": 37.487, "lon": -74.508, "cat": "canyon", "t": "100-fathom tip; southern run"},
        {"n": "Poor Man's Canyon", "lat": 38.05, "lon": -74.03, "cat": "canyon", "t": "Closest canyon (~53 nm)", "approx": True},
        {"n": "Baltimore Canyon", "lat": 38.245, "lon": -73.843, "cat": "canyon", "t": "100-fathom tip; tuna / marlin / mahi"},
        {"n": "Wilmington Canyon", "lat": 38.388, "lon": -73.540, "cat": "canyon", "t": "500-fathom tip; northern canyon"},
        {"n": "Spencer Canyon", "lat": 38.55, "lon": -73.42, "cat": "canyon", "t": "North of Wilmington", "approx": True},
        {"n": "Lindenkohl Canyon", "lat": 38.65, "lon": -73.30, "cat": "canyon", "t": "Northern canyon", "approx": True},
        {"n": "Carteret Canyon", "lat": 38.80, "lon": -73.10, "cat": "canyon", "t": "Far northern run", "approx": True},
    ],
}


@app.get("/spots")
def spots(region: str = Query(None)):
    """Named grounds + the legend. ?region=stuart|oceancity, or all regions."""
    data = {region: SPOTS[region]} if region in SPOTS else SPOTS
    return {"categories": SPOT_CATS, "regions": data}


def _depth_at(lat, lon):
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


@app.get("/depth")
def depth(lat: float = Query(...), lon: float = Query(...)):
    """Charted depth at a point from NOAA ETOPO1 bathymetry (etopo180)."""
    return _depth_at(lat, lon)


def _dir16(u, v):
    """16-point compass label for the direction a vector (u east, v north) points TO."""
    brg = (math.degrees(math.atan2(u, v)) + 360) % 360
    return ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW",
            "WSW", "W", "WNW", "NW", "NNW"][int((brg + 11.25) // 22.5) % 16]


def _nearest(lats, lons, grid, lat, lon):
    i = int(np.abs(np.asarray(lats) - lat).argmin())
    j = int(np.abs(np.asarray(lons) - lon).argmin())
    val = grid[i, j]
    return float(val) if np.isfinite(val) else None


@app.get("/point")
def point(lat: float = Query(...), lon: float = Query(...), date: str = Query(None)):
    """All the ocean values at a clicked point — the click HUD reads from this."""
    date = date or _today_minus(2)
    d = 0.2
    bbox = clamp_bbox((lat - d, lon - d, lat + d, lon + d))
    out = {"lat": round(lat, 4), "lon": round(lon, 4)}

    la, lo, sst, _ = get_field("sst", date, bbox)
    v = _nearest(la, lo, sst, lat, lon)
    if v is not None:
        out["sst_f"] = round(v * 9 / 5 + 32, 1)          # get_field returns °C
    la, lo, chl, _ = get_field("chl", date, bbox)
    v = _nearest(la, lo, chl, lat, lon)
    if v is not None:
        out["chl_mg_m3"] = round(v, 3)

    a = get_altimetry(bbox)
    if a is not None:
        v = _nearest(a["lats"], a["lons"], a["sla"], lat, lon)
        if v is not None:
            out["ssh_m"] = round(v, 3)
            out["eddy"] = "warm-core" if v > 0.05 else "cold-core" if v < -0.05 else "neutral"
        uu = _nearest(a["lats"], a["lons"], a["ugos"], lat, lon)
        vv = _nearest(a["lats"], a["lons"], a["vgos"], lat, lon)
        if uu is not None and vv is not None:
            out["current_kt"] = round(math.hypot(uu, vv) * 1.94384, 2)
            out["current_toward"] = _dir16(uu, vv)

    dw, _ = get_wind(bbox)
    uu = _nearest(dw["lats"], dw["lons"], dw["u"], lat, lon)
    vv = _nearest(dw["lats"], dw["lons"], dw["v"], lat, lon)
    if uu is not None and vv is not None:
        out["wind_kt"] = round(math.hypot(uu, vv) * 1.94384, 1)
        out["wind_from"] = _dir16(-uu, -vv)              # meteorological "from"
    de, _ = get_ekman(bbox)
    uu = _nearest(de["lats"], de["lons"], de["u"], lat, lon)
    vv = _nearest(de["lats"], de["lons"], de["v"], lat, lon)
    if uu is not None and vv is not None:
        out["ekman_kt"] = round(math.hypot(uu, vv) * 1.94384, 2)
        out["ekman_toward"] = _dir16(uu, vv)
    ula, ulo, up, _ = get_upwelling(bbox)
    v = _nearest(ula, ulo, up, lat, lon)
    if v is not None:
        out["upwelling"] = "upwelling" if v > 2e-6 else "downwelling" if v < -2e-6 else "neutral"

    out["depth"] = _depth_at(lat, lon)
    return out


if __name__ == "__main__":
    # Offline self-test: prove every renderer works with no network.
    for name, png in [
        ("selftest_composite.png", composite_png("2000-01-01").body),
        ("selftest_sst.png", sst_png("2000-01-01").body),
        ("selftest_chl.png", chl_png("2000-01-01").body),
        ("selftest_ssh.png", ssh_png("2000-01-01").body),
        ("selftest_upwelling.png", upwelling_png("2000-01-01").body),
    ]:
        open(name, "wb").write(png)
        print(f"OK {name} ({len(png)} bytes)")
    for reg, bb in [("FL", (26.6, -80.3, 27.9, -79.2)), ("MD", (37.2, -75.1, 38.6, -73.3))]:
        s, w, n, e = bb
        cur = currents_json("2000-01-01", s, w, n, e)
        ekm = ekman_json("2000-01-01", s, w, n, e)
        wnd = wind_json("2000-01-01", s, w, n, e)
        print(f"OK currents/ekman/wind [{reg}] "
              f"{len(cur[0]['data'])}/{len(ekm[0]['data'])}/{len(wnd[0]['data'])} cells")
        rep = build_report("2000-01-01", bb)
        print(f"OK report [{reg}] index={rep['index']} grade={rep['grade']} "
              f"strong={rep['strong']} bullets={len(rep['bullets'])}")
