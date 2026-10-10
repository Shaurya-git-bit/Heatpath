from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
from matplotlib import colors
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="HeatPath", page_icon="🔥", layout="wide")

DATA = Path(__file__).parent

ROUTING = {
    "Gorakhpur": dict(key="gorakhpur", bbox=[83.34, 26.73, 83.42, 26.79],
                      orig=(83.365, 26.760), dest=(83.400, 26.745)),
    "Kolkata":   dict(key="kolkata", bbox=[88.32, 22.50, 88.42, 22.60],
                      orig=(88.351, 22.572), dest=(88.388, 22.545)),
}
MAP_ONLY = ["Ahmedabad", "Bikaner", "Raipur"]
LAMBDAS = [1, 3, 10, 30, 100]
WT = {lam: f"c{j}" for j, lam in enumerate(LAMBDAS, 1)}
MAX_DETOUR = 0.25


# ---------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading city data...")
def load_routing(key):
    with open(DATA / f"{key}_roads_scored.pkl", "rb") as f:
        G = pickle.load(f)
    hg = pd.read_parquet(DATA / f"heat_grid_{key}.parquet")
    ids = np.array(list(G.nodes))
    xs = np.array([G.nodes[n]["x"] for n in ids])
    ys = np.array([G.nodes[n]["y"] for n in ids])
    return G, hg, (ids, xs, ys)


@st.cache_data(show_spinner=False)
def load_grid(key):
    return pd.read_parquet(DATA / f"heat_grid_{key}.parquet")


def nearest(nodes, lon, lat):
    ids, xs, ys = nodes
    k = np.cos(np.radians(lat))
    return int(ids[np.argmin(((xs - lon) * k) ** 2 + (ys - lat) ** 2)])


@st.cache_data(show_spinner=False)
def random_trips(key, n=6, min_km=3, max_km=7, seed=42):
    G, _, _ = load_routing(key)
    rng = np.random.default_rng(seed)
    nodes, trips = list(G.nodes), []
    while len(trips) < n:
        o = nodes[rng.integers(len(nodes))]
        d = nx.single_source_dijkstra_path_length(G, o, cutoff=max_km * 1000, weight="length")
        c = [v for v, l in d.items() if l >= min_km * 1000]
        if c:
            trips.append((o, c[rng.integers(len(c))]))
    return trips


# ------------------------------------------------------------- routing
def best_edge(G, a, b, wt):
    return min(G[a][b].values(), key=lambda x: x[wt])


def evaluate(G, src, dst, wt):
    nodes = nx.shortest_path(G, src, dst, weight=wt)
    t = E = TH = L = 0.0
    for a, b in zip(nodes[:-1], nodes[1:]):
        e = best_edge(G, a, b, wt)
        t += e["t"]; E += e["E"]; TH += e["th"]; L += e["length"]
    return nodes, dict(minutes=t, km=L / 1000, exposure=E, mean_temp=TH / t)


def route_xy(G, nodes, wt):
    X, Y = [], []
    for a, b in zip(nodes[:-1], nodes[1:]):
        e = best_edge(G, a, b, wt)
        if "geometry" in e:
            gx, gy = map(list, e["geometry"].xy)
            ax_, ay_ = G.nodes[a]["x"], G.nodes[a]["y"]
            if (gx[0] - ax_) ** 2 + (gy[0] - ay_) ** 2 > (gx[-1] - ax_) ** 2 + (gy[-1] - ay_) ** 2:
                gx, gy = gx[::-1], gy[::-1]
        else:
            gx, gy = [G.nodes[a]["x"], G.nodes[b]["x"]], [G.nodes[a]["y"], G.nodes[b]["y"]]
        X += gx; Y += gy
    return X, Y


# ---------------------------------------------------------------- maps
def heat_layer(hg):
    d = hg.dropna(subset=["heat_anom"])[["lon", "lat", "heat_anom"]].copy()
    h = d["heat_anom"].to_numpy()
    lo, hi = np.percentile(h, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    rgba = matplotlib.colormaps["RdYlBu_r"](colors.Normalize(lo, hi, clip=True)(h)) * 255
    d["color"] = [[int(r), int(g), int(b), 150] for r, g, b, _ in rgba]
    d["heat"] = np.round(h, 2)
    layer = pdk.Layer("ScatterplotLayer", d[["lon", "lat", "color", "heat"]],
                      get_position="[lon, lat]", get_fill_color="color",
                      get_radius=55, radius_min_pixels=1, pickable=True)
    return layer, float(lo), float(hi)


def path_layer(X, Y, color):
    return pdk.Layer("PathLayer", [{"path": [[float(x), float(y)] for x, y in zip(X, Y)]}],
                     get_path="path", get_color=color, width_min_pixels=4)


def point_layer(lon, lat, color):
    return pdk.Layer("ScatterplotLayer", [{"lon": float(lon), "lat": float(lat)}],
                     get_position="[lon, lat]", get_fill_color=color,
                     get_radius=90, radius_min_pixels=7)


def zoom_for(w, s, e, n):
    span = max((e - w) * np.cos(np.radians((s + n) / 2)), n - s)
    return float(np.clip(np.log2(700 * 360 / (256 * span)), 9, 14))


def show_map(layers, center_lon, center_lat, zoom):
    view = pdk.ViewState(latitude=float(center_lat), longitude=float(center_lon), zoom=zoom)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="light",
                             tooltip={"text": "Heat vs city average: {heat} °C"}),
                    use_container_width=True)


# --------------------------------------------------------------- views
def map_only_view(city):
    try:
        hg = load_grid(city.lower())
    except FileNotFoundError:
        st.error(f"Missing data/heat_grid_{city.lower()}.parquet")
        return
    layer, lo, hi = heat_layer(hg)
    w, e = hg.lon.min(), hg.lon.max()
    s, n = hg.lat.min(), hg.lat.max()
    show_map([layer], (w + e) / 2, (s + n) / 2, zoom_for(w, s, e, n))
    st.caption(f"Red = relatively hotter, blue = relatively cooler than the {city} average "
               f"(colour scale {lo:.1f} to {hi:.1f} °C). Hover a cell for its value.")
    st.info("Use this view to pick relatively cooler places to meet, play or wait in summer, "
            "and to spot hot areas where planners could add shade or water points. "
            "It shows the mid-morning pattern, ranked against the city average. "
            "Routing is built for Gorakhpur and Kolkata only.")


def routing_view(city):
    cfg = ROUTING[city]
    try:
        G, hg, nodes = load_routing(cfg["key"])
    except FileNotFoundError:
        st.error(f"Missing data files for {city} in data/")
        return

    trips = [(nearest(nodes, *cfg["orig"]), nearest(nodes, *cfg["dest"]))] + random_trips(cfg["key"])
    labels = ["Demo trip"] + [f"Random trip {i} (3-7 km)" for i in range(1, len(trips))]
    c1, c2 = st.columns(2)
    ti = c1.selectbox("Trip", range(len(trips)), format_func=lambda i: labels[i])
    lam = c2.selectbox("Cooling weight λ", LAMBDAS, index=1,
                       help="Higher λ avoids hot streets more strongly. A route over the 25% "
                            "detour cap falls back to the fastest route.")

    src, dst = trips[ti]
    nodes_f, base = evaluate(G, src, dst, "c0")
    nodes_c, m = evaluate(G, src, dst, WT[lam])
    extra_pct = 100 * (m["minutes"] / base["minutes"] - 1)
    within_cap = extra_pct <= 100 * MAX_DETOUR + 1e-9

    if not within_cap:
        st.warning(f"λ={lam} needs +{extra_pct:.0f}% time, over the 25% cap. Showing the fastest route.")
        nodes_c, m = nodes_f, base
    elif nodes_c == nodes_f:
        st.info("No cooler route exists for this trip at this λ. The fastest route is shown.")

    cut = 100 * (1 - m["exposure"] / base["exposure"]) if base["exposure"] > 0 else 0.0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Extra time", f"+{m['minutes'] - base['minutes']:.1f} min")
    k2.metric("Heat exposure cut", f"{cut:.0f}%")
    k3.metric("Cooler by (avg street heat)", f"{base['mean_temp'] - m['mean_temp']:.2f} °C")
    k4.metric("Route length", f"{m['km']:.1f} km")

    w, s, e, n = cfg["bbox"]
    view = hg[hg.lon.between(w, e) & hg.lat.between(s, n)]
    layer, lo, hi = heat_layer(view)
    fx, fy = route_xy(G, nodes_f, "c0")
    layers = [layer, path_layer(fx, fy, [30, 30, 30])]
    if nodes_c != nodes_f:
        cx, cy = route_xy(G, nodes_c, WT[lam])
        layers.append(path_layer(cx, cy, [255, 0, 255]))
    layers += [point_layer(G.nodes[src]["x"], G.nodes[src]["y"], [0, 160, 0]),
               point_layer(G.nodes[dst]["x"], G.nodes[dst]["y"], [0, 0, 0])]
    show_map(layers, (w + e) / 2, (s + n) / 2, zoom_for(w, s, e, n))
    st.caption("Dark line: fastest route. Magenta line: cooler route. Green dot: start. Black dot: end. "
               f"Red cells are relatively hotter, blue cooler (scale {lo:.1f} to {hi:.1f} °C vs city average). "
               "Bike at 15 km/h, 25% detour cap. Exposure is scored on the model's own predicted heat.")


def about_view():
    st.markdown("""
**HeatPath predicts which 100 m cells of a city run hotter or cooler than the city average, using only free satellite data and no local sensors.**

A LightGBM model is trained on 9 cities (Delhi, Prayagraj, Jaipur, Lucknow, Jamshedpur, Lahore, Karachi, Dhaka, Bangkok) from Landsat 8/9 surface temperature, vegetation, built-up and water indices, and ESA WorldCover land cover at 100 m to 1 km scales. It is then applied to cities it never saw.

**Hot/cool ranking on unseen cities (Spearman)**

| City | Spearman | R² | MAE |
|---|---|---|---|
| Gorakhpur | 0.72 | 0.50-0.70 | 0.91-1.67 °C |
| Kolkata | 0.76 | 0.66-0.70 | 1.27-1.42 °C |
| Ahmedabad | 0.42-0.46 | 0.40-0.43 | 1.33-1.71 °C |
| Bikaner | 0.83 | 0.57 | 1.85 °C |
| Raipur | 0.66 | 0.47 | 1.16 °C |

**Cooler routes (100 random 3-7 km bike trips per city, λ=3, 25% detour cap)**

| City | Median exposure cut (95% CI) | Pooled cut | Within cap | Median extra time |
|---|---|---|---|---|
| Kolkata | 20.5% (16.3-32.1) | 24.9% | 82% | +1.2 min |
| Gorakhpur | 39.5% (33.8-48.2) | 43.6% | 100% | +0.8 min |

**Limits**
- The map shows the persistent mid-morning heat pattern (Landsat overpass), not a forecast or feels-like temperature.
- It ranks places against the city average. Differences are a few °C, so read it as relatively cooler or hotter.
- Exposure cuts are scored on the model's own predicted heat, so they are partly circular.
- The air-temperature difference between routes is small (median 0.2 °C Kolkata, 0.07 °C Gorakhpur). The gain is in exposure to the hottest streets.
- Accuracy varies by city; Ahmedabad is the weakest. Routing covers roughly an 8 x 6.6 km core area per city.
""")


# ----------------------------------------------------------------- app
st.title("HeatPath")
st.markdown("**A street-level heat map for any city, built from free satellite data.** "
            "Find relatively cooler places in 5 cities, with cooler bike routes in Gorakhpur and Kolkata.")

city = st.selectbox("City", list(ROUTING) + MAP_ONLY)
tab_map, tab_about = st.tabs(["Map", "About and results"])

with tab_map:
    if city in ROUTING:
        routing_view(city)
    else:
        map_only_view(city)

with tab_about:
    about_view()