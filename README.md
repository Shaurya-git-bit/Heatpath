# HeatPath

**Know where the heat is, street by street, without a single sensor.**
A street-level heat map for any city, built entirely from free satellite data, with cooler bike routes on top.

**Track:** AI + Climate (prepare for climate impacts) · **ForgeHacks 2026**
**Live demo:** [Streamlit link] · **Demo video:** [[YouTube link]](https://youtu.be/NvK7KZ3ni34) · **Team:** Shaurya Pandey (Solo)

![HeatPath app](images/app_screenshot.png)
*[PLACEHOLDER: screenshot of the Streamlit app showing a heat map with the fastest (dark) and cooler (magenta) route.]*

## The problem
Heat is one of the most immediate climate impacts on city life, but cities like Gorakhpur, Kolkata and Ahmedabad have no street-level heat data. Sensors are sparse, and satellite images are too coarse and too cloud-limited to use directly. People still have to decide where to meet, walk, ride or wait outdoors, and planners have to decide where shade and water points matter most.

## What HeatPath does
Given only free satellite land-cover data, HeatPath predicts which 100 m cells of a city run hotter or cooler than the city average, then uses that map in two ways:
- **Heat map (any city):** see which parts of a city are relatively cooler or hotter. Students can pick a cooler place to meet in summer, and planners can see where shade and water points are needed most.
- **Cooler routes (Gorakhpur and Kolkata):** bike routes that avoid the hottest streets within a 25% detour cap.

## Results at a glance
- Ranks hot and cool spots in **5 cities the model never saw**: Spearman **0.42 to 0.83** (Gorakhpur 0.72, Kolkata 0.76, Bikaner 0.83).
- Cooler routes cut heat exposure by a **median 20.5% in Kolkata and 39.5% in Gorakhpur** for about a minute of extra riding (λ=3, 100 random trips per city).
- **93% of Gorakhpur trips** get a cooler route; the cap keeps every route within +25% travel time.
- No local sensors, no paid data. A new city needs only a bounding box.

## How it works
**Data (all free).**
- Landsat 8/9 Collection 2 Level 2: land surface temperature (`lst_c`), plus NDVI, NDBI and NDWI. Scenes have under 20% cloud, with cloud, shadow and saturated pixels masked.
- ESA WorldCover v200: tree, built-up, water and low-vegetation cover within 100 m, plus neighbourhood context (built-up at 300 m and 1 km, trees at 300 m, water at 500 m).
- SRTM elevation, and OpenStreetMap roads via osmnx.

**Training set.** 3,000 sampled points per city per Landsat scene, 2024 to 2026, from 9 cities: Delhi, Prayagraj, Jaipur, Lucknow, Jamshedpur, Lahore, Karachi, Dhaka, Bangkok.

**Model.** LightGBM regressor (400 trees, learning rate 0.05, 31 leaves, min 50 samples per leaf, 0.8 row and column subsampling).
- **Target:** the *anomaly*, `lst_c` minus that scene's mean. This removes weather, season and time of day, so one model transfers across cities and days.
- **Features:** the 11 surface features above, with no weather input, so it runs anywhere without local data.
- **Why LightGBM:** land-cover effects are non-linear and interact (built-up vs tree cover vs water), the data is tabular, and training is fast. SHAP makes it explainable.
- **Calibration:** a leave-one-city-out slope gives a scaling factor K = 1.05 (close to 1, so predictions are well calibrated). K comes from training cities only.

**Prediction.** For each 100 m cell the model predicts the anomaly, centres it and applies K. It then averages up to 3 clear scenes (March to June) into one persistent heat grid.

**Routing.** The bike graph (15 km/h) is scored per street segment, sampled every 20 m. Heat excess is the predicted heat above the city median. Exposure = riding time × heat excess, and route cost = time × (1 + λ × heat excess), with λ from 1 to 100. If a route exceeds the 25% detour cap, the rider gets the fastest route (counted as a 0% cut).

**Validation (no leakage).**
1. Block + scene cross-validation (held-out spatial blocks and scenes).
2. Leave-one-city-out on the 9 training cities: R² 0.28 to 0.56 (mean 0.44).
3. **Five unseen cities**, scored against observed Landsat on cells the model never saw. An assertion in the code confirms test cities are not in the training set.

## Results on unseen cities
| City | Scenes | Spearman | R² | MAE (°C) |
|---|---|---|---|---|
| Bikaner | pooled | **0.83** | 0.57 | 1.85 |
| Kolkata | 3 | **0.76** pooled (0.72 to 0.76 per scene) | 0.66 to 0.70 | 1.27 to 1.42 |
| Gorakhpur | 3 | **0.72** pooled (0.60 to 0.77 per scene) | 0.50 to 0.70 | 0.91 to 1.67 |
| Raipur | pooled | **0.66** | 0.47 | 1.16 |
| Ahmedabad | 3 | 0.42 to 0.46 per scene | 0.40 to 0.43 | 1.33 to 1.71 |

*Spearman ranks the hot-to-cool order of 100 m cells within a city. R² and MAE are per scene for Gorakhpur, Kolkata and Ahmedabad, and pooled for Bikaner and Raipur.*

<img width="1190" height="1072" alt="bikaner_raipur" src="https://github.com/user-attachments/assets/e99b9a2b-38f0-468c-8a99-9b02ab7b275f" />


*Bikaner (top) and Raipur (bottom), two cities the model never saw: predicted (left) vs observed Landsat (right), °C vs scene mean, first usable scene shown. In Bikaner the model places the cool core and both hot zones (north and southeast) where Landsat sees them (Spearman 0.83 over 3 scenes). In Raipur it finds the cooler river corridor and the hot edges (Spearman 0.66 over 3 scenes). It softens the extremes, so read the map as a ranking of relatively hotter and cooler places.*

<img width="756" height="420" alt="shap_glo" src="https://github.com/user-attachments/assets/b76d48b1-426f-4b6d-a783-8e6f0740d5ed" />

*SHAP for the full model on absolute temperature. Sun elevation at the overpass dominates the temperature level (mean |SHAP| about 4.9 °C), followed by NDBI, air temperature and elevation. Weather sets how hot a scene is; surfaces decide which places are hotter within it. That is why HeatPath predicts the anomaly from surfaces alone.*

<img width="789" height="739" alt="shap_mea" src="https://github.com/user-attachments/assets/0460142f-a666-4619-bf0b-12b3e15433a3" />

*SHAP for the spatial model (7-feature version; the shipped model adds 4 neighbourhood context features). A high built-up index (NDBI) and built-up cover push cells hotter, by up to about 4 °C. Tree cover, water (NDWI, water cover) and vegetation push them cooler.*

## Cooler routes: results
100 random 3 to 7 km bike trips per city, 25% detour cap, trips over the cap fall back to the fastest route.

| City | λ | Median exposure cut (95% CI) | Pooled cut | Trips with a cooler route | Within cap | Median extra time |
|---|---|---|---|---|---|---|
| Kolkata | 1 | 20.4% (16.8 to 24.9) | 24.4% | 91% | 100% | +0.6 min |
| Kolkata | **3** | **20.5%** (16.3 to 32.1) | 24.9% | 79% | 82% | +1.2 min |
| Kolkata | 10 | 13.2% (0.0 to 23.0) | 18.3% | 55% | 56% | +0.7 min |
| Gorakhpur | 1 | 12.5% (7.6 to 20.1) | 23.2% | 71% | 100% | +0.1 min |
| Gorakhpur | **3** | **39.5%** (33.8 to 48.2) | 43.6% | 93% | 100% | +0.8 min |
| Gorakhpur | 10 | 55.9% (49.5 to 59.4) | 51.5% | 91% | 92% | +2.3 min |

λ=3 is the headline setting for both cities. A higher λ cuts more where the cap allows it (Gorakhpur λ=10) but hits the cap more often (Kolkata λ=10). Single demo trips: Kolkata λ=10, +1.7 min (+7%), 40% less exposure; Gorakhpur λ=3, +1.7 min (+8%), 43% less exposure.

## Impact
- **Residents and students:** pick a relatively cooler place to meet, play or wait; ride, walk or deliver along cooler streets for a minute or so of extra time.
- **City planners:** street-by-street hot spots show where shade trees, water points and cooling shelters will help most, in cities that had no such map.
- **Scales cheaply:** Landsat and WorldCover cover the whole planet and cost nothing, so a new city is a bounding box away. Accuracy has been tested on 5 unseen South Asian cities, and other climates are untested.
- **Next steps:** validate against ground sensors, add afternoon and evening passes, add feels-like using weather, and plug the heat layer into transit, delivery and cycling apps.

## Limits
- The map shows the persistent **mid-morning** pattern (Landsat overpass), not a forecast or feels-like temperature.
- It ranks places against the city average. Differences are a few °C, so read it as relatively cooler or hotter.
- Route exposure is scored on the model's own predicted heat, so the cuts are partly circular. The temperature difference between routes is small (median 0.21 °C Kolkata, 0.07 °C Gorakhpur); the gain is in exposure to the hottest streets.
- Ahmedabad is the weakest city (Spearman 0.42 to 0.46). Routing covers about 10 × 11 km in Kolkata and 8 × 6.6 km in Gorakhpur.

## Repo structure
```
heatpath/
├── heatpath.py                       # Streamlit app (map, routing, results)
├── requirements.txt                  # App dependencies
├── README.md
├── lst-climate.ipynb                 # Full pipeline: sampling, training, validation, SHAP, unseen-city tests, routing
└── data/
    ├── heatpath_raw_v3.parquet       # Training data (9 cities, Landsat + land-cover features)
    ├── heat_grid_gorakhpur.parquet   # Predicted heat grids for the 5 unseen cities
    ├── heat_grid_kolkata.parquet
    ├── heat_grid_ahmedabad.parquet
    ├── heat_grid_bikaner.parquet
    ├── heat_grid_raipur.parquet
    ├── gorakhpur_roads_scored.pkl    # Bike road graphs scored with predicted heat
    └── kolkata_roads_scored.pkl
```

## Run it
```bash
pip install -r requirements.txt
streamlit run heatpath.py
```
The notebook runs on Kaggle with an Earth Engine project. To rerun it locally, point the `pd.read_parquet` path at `data/heatpath_raw_v3.parquet`.

**Data:** Landsat 8/9 Collection 2 Level 2, ESA WorldCover v200, SRTM, OpenStreetMap, Open-Meteo.
