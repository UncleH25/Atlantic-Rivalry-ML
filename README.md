# Atlantic Rivalry
### Machine Learning Analysis of the White Star Line and Cunard Line

*A complete data science project — from web scraping to SHAP explainability — using real historical ship records spanning 1836 to 1960.*

---

## The Historical Question

For nearly a century, two British shipping companies fought one of the most consequential commercial rivalries in history. The **White Star Line** and the **Cunard Line** competed for dominance of the North Atlantic passenger trade — the most profitable and prestigious route in the world. Between them they built some of the most famous ships ever to sail: the Titanic, the Lusitania, the Mauretania, the Olympic, the Queen Mary. Their competition drove ships to grow from small wooden sailing vessels of a few hundred tons to ocean liners exceeding 80,000 tons within a single human lifetime.

This project is not about the Titanic disaster specifically. It is about the full picture of both fleets across 120 years: what the data reveals about the rivalry, what actually caused ships to be lost, and whether machine learning can detect the difference between the two companies from nothing but numbers.

---

## The Data

All data is real historical records drawn from publicly available sources.

**Primary sources:**
- [Wikipedia — List of White Star Line ships](https://en.wikipedia.org/wiki/List_of_White_Star_Line_ships)
- [Wikipedia — List of Cunard Line ships](https://en.wikipedia.org/wiki/List_of_Cunard_Line_ships)

Each article contains structured tables of ships with name, build year, gross register tonnage (GRT), years in service, and notes on fate. The data was scraped programmatically using `requests` and `BeautifulSoup`, then cleaned from raw Wikipedia format into ML-ready features.

**Dataset after cleaning:**
- 326 ships total (44 post-1960 vessels removed as outside the rivalry era)
- 134 White Star Line ships, 192 Cunard Line ships
- Build years spanning 1836 to 1960
- GRT ranging from 6 tons to 83,673 tons
- 115 ships lost at sea, 206 survived their working lives, 5 with unknown fate

---

## Project Structure

The project runs as a single Google Colab notebook — `Atlantic_Rivalry.ipynb` — organised into nine sequential phases.

### Phase 1 — Data Collection
Scraping Wikipedia tables for both shipping lines using `requests` and `BeautifulSoup`. Combining 134 + 192 ships into a single dataset with a `line` identifier column. Saving a raw CSV checkpoint before any cleaning.

**Notable debug:** Wikipedia requires a User-Agent header in HTTP requests. Without it, every request returns a 403 Forbidden error. Solved by adding a descriptive `User-Agent` string identifying the project.

### Phase 2 — Data Cleaning
Real Wikipedia data is messy. GRT values contained footnote markers (`[2]`), comma separators, and multiple measurements for ships that were remeasured after wartime refits (e.g. Queen Mary: `80,774 (1936) 81,237 (1947)`). Build years contained unknown values (`18??`). Dates were stored as ranges (`1852–1868`).

The most consequential cleaning step was creating `is_lost` — a binary target variable derived from free-text Notes entries using keyword classification. This required three iterations:

- **v1:** Basic keywords (`wrecked`, `torpedoed`, `mined`)
- **v2:** Added U-boat designation patterns (`SM U-53`, `u-\d`, `UC-`, `UB-`) after discovering 25 submarine sinkings were landing in the Unknown bucket because Wikipedia uses formal German Navy designations rather than plain English
- **v3:** Added `u boat` (space variant), `fleet air`, and `iceberg` (correctly classifying the Titanic as a collision)

The debugging workflow — inspecting the Unknown bucket before trusting any chart — is described in detail in the notebook and is the clearest example in the project of why iterative data validation matters.

### Phase 3 — Exploratory Data Analysis
Six visualisations telling the story of the rivalry before any modelling:

1. Ships built per decade — the two construction peaks and White Star's disappearance after 1934
2. Average tonnage per decade — the arms race in steel from ~1,000 GRT to ~20,000 GRT mean
3. Loss rates and causes (corrected v3) — 42.6% White Star vs 31.2% Cunard; torpedoes dominate
4. Tonnage distribution by era — clear staircase pattern confirming era as a strong ML feature
5. Service life and survival — median 11.5 years for lost ships vs 14.0 for survivors
6. Ship size vs loss probability — a threshold at ~1,824 GRT, then flat regardless of further size

### Phase 4 — Feature Engineering
Fourteen ML-ready features created from raw data:

| Feature | Type | Description |
|---------|------|-------------|
| `grt_clean` | numeric | Cleaned tonnage |
| `year_built`, `year_entered` | numeric | Construction and service start |
| `service_years`, `decade_built` | numeric | Career length and era band |
| `line_encoded` | binary | 0=White Star, 1=Cunard |
| `size_cat_encoded` | ordinal | 0–3 quartile-based size |
| `wartime_flag`, `served_ww1`, `served_ww2` | binary | War exposure |
| `era_Victorian`, `era_WWI`, `era_Interwar`, `era_WWII+` | one-hot | Historical period |

**Notable bug caught and fixed:** The SimpleImputer was detecting which features needed imputation from `X_train` rather than the full `X`. One ship (the Justicia, whose service period was `"Never operated"`) had her single missing value land entirely in the test split, meaning `X_train` showed no NaN in that column and the imputer never learned to handle it. The fix detects missing features from the full dataset while still fitting imputation statistics on training data only.

### Phase 5 — Classification: Predicting Ship Loss
**Question:** Given a ship's specifications and the era it operated in, can a model predict whether it was lost at sea?

| Model | Accuracy | F1 | False Alarms | Missed Losses |
|-------|----------|----|-------------|---------------|
| Logistic Regression | 63.1% | 0.538 | 15 | 9 |
| **Random Forest** | **72.3%** | **0.571** | **7** | **11** |
| XGBoost | 66.2% | 0.542 | 12 | 10 |

Baseline (always predict survived): 64.6%. Random Forest beats it by +7.7 percentage points.

**Caveat:** `service_years` ranked as the top feature by importance. This feature is partially circular — ships that sank had shorter careers because they sank. It is not outright data leakage (it was not known in advance), but its high importance should be interpreted cautiously. Phase 9's SHAP analysis examines this in detail.

### Phase 6 — Regression: Predicting Ship Tonnage
**Question:** Can we predict how large a ship was from when it was built and who operated it?

| Model | R² | MAE |
|-------|-----|-----|
| Linear Regression | 0.249 | 4,893 tons |
| **Ridge Regression** | **0.258** | **4,857 tons** |
| Gradient Boosting | 0.137 | 5,066 tons |

The best model explains 25.8% of tonnage variation. The remaining 74% reflects individual strategic decisions — a company might build a giant prestige liner in the same decade as a modest cargo vessel. The era sets the ceiling; ambition and economics determine where each ship lands within it.

### Phase 7 — White Star or Cunard?
**Question:** Given only ship specifications, can a model identify which of the two lines a ship belonged to?

- Test accuracy: **69.7%** (+10.6pp over 59.1% baseline)
- Stratified 5-fold CV: **77.3% ± 0.052**

**Notable diagnostic:** The initial cross-validation produced scores of `[0.682, 0.569, 0.292, 0.308, 0.600]` with mean 0.490 — below baseline. Investigation revealed that unstratified folds on a temporally structured dataset can produce wildly unbalanced class distributions. Fixing to `StratifiedKFold` gave stable results (0.773 ± 0.052) and confirmed the original test score was trustworthy.

**Historical answer:** Yes, the two fleets were detectably different — but only modestly. The model is primarily learning that Cunard operated across more eras and that White Star built proportionally larger ships in the 1890–1920 period. In the Victorian emigrant trade, the two lines were statistically indistinguishable from their ship numbers alone.

### Phase 8 — Clustering: What Types of Ships Were There?
K-Means unsupervised clustering (k=10, selected by silhouette score) found ten distinct ship groupings without being told anything about company names, eras, or ship types.

The most unexpected finding: **two clusters share identical era and size profiles but are 100% company-pure**. Cluster 1 is 100% White Star Victorian small vessels; Cluster 3 is 100% Cunard Victorian small vessels. The algorithm detected a real numerical difference between the two companies' early fleets without any company label.

Their loss rates diverge sharply: 47.5% for White Star vs 25.3% for Cunard in the Victorian era — almost certainly reflecting different route choices. White Star's early ships ran to Australia via Cape Horn, one of the most dangerous passages on earth. Cunard's focus was the North Atlantic.

**Cluster 8 (five ships, all built 1914) had a 60% loss rate** — the worst in the dataset. Majestic, Britannic, Aquitania, Justicia, Belgic. Three of the five were lost. The year 1914 was the worst possible moment to launch a giant liner.

### Phase 9 — SHAP Explainability
SHAP (SHapley Additive exPlanations) values decompose every prediction from the Random Forest model into the exact contribution of each feature, ship by ship.

**Global findings (mean |SHAP| across all 321 ships):**

| Rank | Feature | Mean |SHAP| | Category |
|------|---------|-------------|----------|
| 1 | service_years | 0.0870 | Temporal |
| 2 | year_entered | 0.0667 | Temporal |
| 3 | grt_clean | 0.0602 | Size |
| 4 | year_built | 0.0425 | Temporal |
| 5 | line_encoded | 0.0349 | Company |

**Individual ship predictions:**
- **Titanic** — 90.5% loss probability. Every feature pushed toward loss: WWI/WWII exposure, 1912 build year, large size, wartime era.
- **Lusitania** — 80.0% loss probability. Same pattern, slightly lower because her 1907 launch gave the model a marginally longer projected career.
- **Olympic** — Predicted LOST (64% probability), actually SURVIVED. The model sees Titanic's near-identical sister — same year, same size, same company, same wartime exposure — and predicts loss. Only `service_years` (24 years, a long career) pulls toward survival. The model is not wrong: the Olympic beat enormous historical odds. Her 64% loss prediction is an expression of genuine uncertainty about a ship that survived when her sisters did not.
- **Mauretania** — 8% loss probability. A 27-year career and large GRT both push strongly toward survival. Correct — she was scrapped peacefully in 1935.

**The GRT dependence plot** confirms the 1,824-ton threshold with clarity: SHAP values for `grt_clean` are scattered and significant for ships below the threshold, then collapse toward zero above it. Size stops predicting loss past that point. What killed ships was when they sailed, not how large they were.

---

## Key Findings

**1. Wartime exposure was the dominant predictor of ship loss.** The two world wars killed more ships than all peacetime causes combined. Torpedo losses alone (43 ships) exceeded any other single category. Era and temporal features topped every feature importance chart across all models.

**2. The rivalry was real but narrower than legend suggests.** Both lines were statistically indistinguishable in the Victorian emigrant trade. Measurable differences emerged in specific periods: White Star built proportionally larger ships in 1890–1920; Cunard outlasted them by 25 years and survived the 1934 merger.

**3. The 1914 cohort was the most tragic in maritime history.** Five giants launched that year — three were lost. Ships of that size had never existed before; neither had the weapons designed to sink them.

**4. Small Victorian ships had split fates by company.** White Star's early fleet lost 47.5% of its ships vs 25.3% for Cunard's. Route choices — Cape Horn to Australia vs the North Atlantic — most likely explain the gap.

**5. Ship size predicted loss at one threshold only.** Under 1,824 GRT: 27.2% loss rate. Above it: 38–40% regardless of further size. The threshold separates Victorian-era ships from the modern steam fleet, not large ships from small ones.

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Atlantic_Rivalry.ipynb` or open it directly from GitHub
3. Run all cells from top to bottom (`Runtime → Run all`)
4. Phases 2–9 each reload from saved CSV checkpoints and can be re-run independently after Phase 1 has been completed once

**Dependencies:** All libraries are pre-installed in Google Colab. Two additional packages are installed at the top of the notebook:
```
!pip install xgboost --quiet
!pip install shap --quiet
```

**Runtime:** Approximately 8–12 minutes for a full run, dominated by Phase 8 (GridSearchCV, 360 model fits) and Phase 9 (SHAP computation).

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `requests`, `BeautifulSoup` | Web scraping |
| `pandas`, `numpy` | Data manipulation and cleaning |
| `re` | Regular expressions for data cleaning |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | ML models, preprocessing, evaluation, PCA, K-Means |
| `xgboost` | Gradient boosting classifier |
| `shap` | Model explainability |

---

## Caveats and Limitations

**service_years circularity.** The `service_years` feature ranked first in both Random Forest and SHAP importance. For lost ships, this measures how long they sailed before sinking — partially circular with the target variable. The models are not wrong to use it, but its dominance reflects this structure. Future work could train models with `service_years` excluded to isolate the contribution of genuinely forward-looking features.

**Wikipedia data quality.** The source data contains OCR artifacts (`"Juicy"` for `"Fiji"`), inconsistent fate descriptions, and genuinely missing records. The keyword classifier for `is_lost` required three iterations to correctly classify U-boat sinkings. Five ships remain with unknown fate and are excluded from ML training.

**Small dataset.** 321 ships with known fates is a small dataset for ML. XGBoost underperformed Random Forest in Phase 5 for exactly this reason — its sequential boosting requires more data to generalise. Results should be interpreted as exploratory and historically suggestive rather than statistically definitive.

**Scope decisions.** Ships built after 1960 were excluded as outside the rivalry era. The four modern Cunard ships (Queen Mary 2, Queen Victoria, Queen Elizabeth, Queen Anne) included in the Wikipedia table would distort every tonnage and era calculation if retained. This is a documented scope choice, not data cleaning.

---

## Acknowledgements

Historical ship data sourced from Wikipedia under the Creative Commons Attribution-ShareAlike License. This project is for educational purposes.

---

*Built as a machine learning portfolio project. Nine phases from raw web scraping to SHAP explainability, using real historical data throughout.*
