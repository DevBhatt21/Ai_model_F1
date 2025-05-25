import os
import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ——— Configuration ———
API_KEY      = os.getenv("OPENWEATHER_API_KEY", "")
CACHE_DIR    = "f1_cache"
DEFAULT_TEMP = 22.0  # Expected temperature at Imola
TRACK_NAME   = "Monaco"
SC_CHANCE = 1.0 #100% safety car chance  

# Track-specific overtake factors (0 = very hard to overtake, 1 = easy)
track_overtake_factor = {
    "Monaco": 0.0,
    "Melbourne": 0.13,
    "Imola": 0.17,
    "Saudia arabia": 0.25,
    "Singapore city": 0.26,
    "Budapest": 0.27,
    "Abu Dhabi": 0.28,
    "Barcelona": 0.30,
    "Montreal": 0.30,
    "Zandvoort": 0.32,
    "Silverstone": 0.37,
    "Mexico city": 0.43,
    "Spielberg": 0.47,
    "Monza": 0.55,
    "Austin": 0.57,
    "Spa": 0.62,
    "Suzuka": 0.63,
    "Baku": 0.70,
    "Sakhir": 0.72,
    "Le Castellet": 0.74,
    "Sao Paulo": 0.90,
    "China": 0.95,
    "Bahrain": 1.00,
}
OVERTAKE_FACTOR = track_overtake_factor.get(TRACK_NAME, 0.5)

# Create the cache directory if it doesn't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

fastf1.Cache.enable_cache(CACHE_DIR)

# ——— 1) Load & preprocess 2025 Miami GP data ———
session = fastf1.get_session(2024, "Monaco", "R")
session.load()

laps = (
    session.laps
    .dropna(subset=["LapTime","Sector1Time","Sector2Time","Sector3Time"])
    .copy()
)
# Convert to seconds
laps["LapTime_s"]     = laps["LapTime"].dt.total_seconds()
laps["Sector1Time_s"] = laps["Sector1Time"].dt.total_seconds()
laps["Sector2Time_s"] = laps["Sector2Time"].dt.total_seconds()
laps["Sector3Time_s"] = laps["Sector3Time"].dt.total_seconds()

# Historical average lap time → AvgLapTime_s
y_df = (
    laps.groupby("Driver")["LapTime_s"]
    .mean()
    .reset_index()
    .rename(columns={"LapTime_s": "AvgLapTime_s"})
)

# Average sector times
sector_avgs = (
    laps.groupby("Driver")[["Sector1Time_s","Sector2Time_s","Sector3Time_s"]]
    .mean()
    .reset_index()
)

# ——— 2) This weekend’s qualifying data  ———
qual = pd.DataFrame({
    "Driver": ["PIA","NOR","RUS","VER","LEC","SAI","HAM","ALB","OCO","GAS","TSU","ALO","STR","ANT","LAW"],
    "QualifyingTime (s)": [70.129,69.954,71.507,70.669,70.063,71.362,70.382,71.213,70.942,71.994,71.415,70.924,72.563,71.880,71.129]
})
qual["QualifyingTime"] = qual["QualifyingTime (s)"]
qual["QualRank"] = qual["QualifyingTime"].rank(method="min")
qual["CleanAirBonus"] = (qual["QualRank"].max() - qual["QualRank"]) / (qual["QualRank"].max() - 1)
qual["StartingGridPosition"] = qual["QualRank"]

# ——— 3) Wet‑weather adjustment ———
wet_scores = {
    "VER":0.97979,"NOR":0.991275,"PIA":1.00000,"LEC":1.004386,
    "RUS":0.970804,"HAM":0.952873,"GAS":0.973042,"ALO":0.963240,
    "TSU":0.97000,"SAI":0.998941,"HUL":0.991394,"OCO":0.984206,
    "STR":0.959010,"ANT":1.00000,"ALB":0.988258
}
qual["WetScore"] = qual["Driver"].map(wet_scores).fillna(1.0)

# Weather for Imola
try:
    w = requests.get(
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q=Monaco&appid={API_KEY}&units=metric",
        timeout=5
    ).json()
    temp      = w.get("main",{}).get("temp", DEFAULT_TEMP)
    rain_mm   = w.get("rain",{}).get("1h", 0.0)
    rain_prob = 1.0 if rain_mm>0 else 0.0
except:
    temp, rain_prob = DEFAULT_TEMP, 0.0

print(f"🏁 Monaco GP Weather → Temp: {temp}°C, Rain Prob: {rain_prob}")

if rain_prob > 0.75:
    qual["QualifyingTime"] *= qual["WetScore"]

# ——— 4) Team performance ———
driver_to_team = {
    "VER":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","RUS":"Mercedes",
    "HAM":"Ferrari","GAS":"Alpine","ALO":"Aston Martin","TSU":"Racing Bulls",
    "SAI":"Ferrari","HUL":"Kick Sauber","OCO":"Alpine","STR":"Aston Martin",
    "ANT":"Mercedes","ALB":"Williams"
}
team_points = {
    "McLaren":279,"Mercedes":147,"Red Bull":131,"Ferrari":114,"Williams":51,
    "Aston Martin":14,"Racing Bulls":10,"Alpine":7,"Kick Sauber":6
}
maxp = max(team_points.values())
team_perf = {t: pts/maxp for t,pts in team_points.items()}

qual["Team"]            = qual["Driver"].map(driver_to_team)
qual["TeamPerformance"] = qual["Team"].map(team_perf).fillna(np.mean(list(team_perf.values())))

# ——— 5) Custom Feature Engineering ———
# UpgradeEffectiveness
upgrade_boost = {
    "Ferrari": 0.8, "Mercedes": 0.3, "McLaren": 0.5, "Red Bull": 0.8, "Aston Martin": 0.4, "Alpine": 0.5, "Racing Bulls": 0.0, "Williams": 0.3, "Kick Sauber": 0.3, "Alfa Romeo": 0.0
}
qual["UpgradeEffectiveness"] = qual["Team"].map(upgrade_boost).fillna(0.0)

# PitStrategyScore
strategy_score = {
    "Ferrari": 0.8, "Mercedes": 1.0, "Red Bull": 0.95, "McLaren": 0.9,
    "Alpine": 0.7, "Aston Martin": 0.6, "Racing Bulls": 0.5, "Williams": 0.5, "Kick Sauber": 0.6, "Alfa Romeo": 0.5
}
qual["PitStrategyScore"] = qual["Team"].map(strategy_score).fillna(0.6)

# SafetyCarChance — uniform across all
qual["SafetyCarChance"] = SC_CHANCE

# Driver Points (Recent Form)
driver_points = {
    "VER": 124, "NOR": 133, "LEC": 61, "PIA": 146, "SAI": 11,
    "RUS": 99, "HAM": 53, "ALO": 0, "TSU": 10, "STR": 14,
    "ALB": 40, "OCO": 14, "GAS": 7, "ANT": 48, "HUL": 6, "Had": 7, "Bearman": 6, "Law": 0
}
max_driver_pts = max(driver_points.values())
qual["DriverPoints"] = qual["Driver"].map(driver_points).fillna(0) / max_driver_pts

# ——— 6) Assemble final DataFrame ———
df = (
    qual
    .merge(sector_avgs, on="Driver", how="left")
    .merge(y_df,      on="Driver", how="left")
    .assign(Temperature=temp, RainProb=rain_prob, OvertakeFactor=OVERTAKE_FACTOR)
    .dropna(subset=["AvgLapTime_s"])
)

# Features & target
X = df[[ 
    "QualifyingTime",
    "Sector1Time_s", "Sector2Time_s", "Sector3Time_s",
    "Temperature", "RainProb", "TeamPerformance", "OvertakeFactor", "CleanAirBonus", "UpgradeEffectiveness", "PitStrategyScore", "SafetyCarChance", "DriverPoints", "StartingGridPosition"
]]
y = df["AvgLapTime_s"]

# ——— 7) Train/test split & model ———
X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = make_pipeline(
    SimpleImputer(strategy="mean"),
    GradientBoostingRegressor(n_estimators=250, learning_rate=0.1, random_state=42)
)
model.fit(X_train, y_train)

# ——— 8) Predict & evaluate ———
df["PredLapTime_s"] = model.predict(X)
print("\nPredicted 2025 Monaco GP Order:")
print(df.sort_values("PredLapTime_s")[["Driver","PredLapTime_s"]].to_string(index=False))
print(f"\nTest MAE: {mean_absolute_error(y_test, model.predict(X_test)):.3f} s")

# ——— 9) Plots ———
# a) Qual vs Pred
plt.figure(figsize=(8,6))
plt.scatter(df["QualifyingTime"], df["PredLapTime_s"], c="C0")
minq, maxq = df["QualifyingTime"].min(), df["QualifyingTime"].max()
plt.plot([minq,maxq],[minq,maxq],"r--")
for _,r in df.iterrows():
    plt.annotate(r.Driver, (r.QualifyingTime,r.PredLapTime_s),
                 textcoords="offset points", xytext=(5,5), fontsize=8)
plt.xlabel("Qualifying Time (s)")
plt.ylabel("Predicted Lap Time (s)")
plt.title("Qualifying vs Predicted Lap")
plt.tight_layout()
plt.show()

# b) Feature importances
fi = model.named_steps["gradientboostingregressor"].feature_importances_
feat_df = pd.DataFrame({
    "Feature": X.columns, "Importance": fi
}).sort_values("Importance", ascending=False)
print("\nFeature Importances:")
print(feat_df.to_string(index=False))

plt.figure(figsize=(6,4))
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.tight_layout()
plt.show()
