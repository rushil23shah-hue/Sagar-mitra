"""
Sagar Mitra Backend - Data Layer
=================================
Loads the historical Argo dataset ONCE at startup (not per-request), builds
the 4D climatology baseline used by the statistical agent, and provides the
synthetic demo profile/historical-timeseries used as a stand-in until a real
per-location Argo profile lookup is implemented.

DATA SOURCE PRIORITY:
  1. A cached Parquet file at DATA_PATH (fast, recommended -- see README).
  2. If missing, falls back to a small synthetic dataset so the API still
     boots and is testable by the frontend team while real data prep is
     finished separately.
"""
import os
import numpy as np
import pandas as pd

DATA_PATH = os.environ.get("SAGAR_MITRA_DATA_PATH", "./data/final_dataset.parquet")

DEPTH_BINS = [0, 25, 75, 150, 300, 600, 1000, 2000]
DEPTH_LABELS = ["0-25m", "25-75m", "75-150m", "150-300m", "300-600m", "600-1000m", "1000-2000m"]


def _make_synthetic_dataset(n_points: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Small synthetic Argo-like dataset so the backend boots without real data."""
    rng = np.random.default_rng(seed)
    lat = rng.uniform(14.0, 20.0, n_points)   # Maharashtra / Arabian Sea coastal band
    lon = rng.uniform(68.0, 74.0, n_points)
    pres = rng.uniform(0, 1800, n_points)
    juld = pd.to_datetime(
        rng.integers(pd.Timestamp("2015-01-01").value // 10**9,
                     pd.Timestamp("2025-12-31").value // 10**9,
                     n_points), unit="s"
    )
    temp = 28 - (pres / 1800) * 20 + rng.normal(0, 0.8, n_points)
    psal = 35 + rng.normal(0, 0.3, n_points)
    df = pd.DataFrame({
        "LATITUDE": lat, "LONGITUDE": lon, "PRES": pres,
        "TEMP": temp, "PSAL": psal, "JULD": juld,
    })
    return df


def _build_demo_profile_and_hist():
    """Synthetic single-cast profile + daily surface timeseries, matching the
    shape ArgoStatisticalAnomalyEngine.analyze_profile expects (from the
    original notebook's self-test block)."""
    rng = np.random.default_rng(7)
    n = 20
    depths = np.linspace(2, 1800, n)
    profile = pd.DataFrame({
        "LATITUDE": 16.99, "LONGITUDE": 73.30,
        "PRES": depths,
        "TEMP": 29.5 - (depths / 1800) * 22 + rng.normal(0, 0.3, n),
        "PSAL": 34.5 + rng.normal(0, 0.15, n),
        "WMO_ID": 2902731, "CYCLE": 1,
        "JULD": pd.Timestamp("2026-08-01"),
    })
    days = pd.date_range("2026-07-01", periods=30, freq="D")
    hist = pd.DataFrame({
        "DATE": days,
        "CT": 27.5 + rng.normal(0, 0.4, len(days)),
    })
    return profile, hist


class DataStore:
    """Holds everything loaded once at app startup."""

    def __init__(self):
        self.final_dataset: pd.DataFrame
        self.climatology_baseline: pd.DataFrame
        self.regional_fallback: pd.DataFrame
        self.demo_profile: pd.DataFrame
        self.demo_hist: pd.DataFrame
        self.using_synthetic_data: bool = False
        self._load()

    def _load(self):
        if os.path.exists(DATA_PATH):
            self.final_dataset = pd.read_parquet(DATA_PATH)
            self.using_synthetic_data = False
        else:
            print(f"[data] WARNING: {DATA_PATH} not found -- using synthetic "
                  f"demo data. Real Argo dataset should be cached there "
                  f"(see README) before the actual hackathon demo.")
            self.final_dataset = _make_synthetic_dataset()
            self.using_synthetic_data = True

        self._build_climatology()
        self.demo_profile, self.demo_hist = _build_demo_profile_and_hist()

    def _build_climatology(self):
        df = self.final_dataset.copy()
        df["JULD"] = pd.to_datetime(df["JULD"])
        df["MONTH"] = df["JULD"].dt.month
        df["LAT_BIN"] = np.floor(df["LATITUDE"] / 2.0) * 2.0
        df["LON_BIN"] = np.floor(df["LONGITUDE"] / 2.0) * 2.0
        df["DEPTH_BIN"] = pd.cut(df["PRES"], bins=DEPTH_BINS, labels=DEPTH_LABELS, right=False)

        self.climatology_baseline = df.groupby(
            ["LAT_BIN", "LON_BIN", "DEPTH_BIN", "MONTH"], observed=False
        ).agg(
            temp_mean=("TEMP", "mean"),
            temp_std=("TEMP", "std"),
            temp_p90=("TEMP", lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) >= 5 else np.nan),
            sal_mean=("PSAL", "mean"),
            sal_std=("PSAL", "std"),
            sample_count=("TEMP", "count"),
        ).reset_index()

        self.regional_fallback = df.groupby(
            ["DEPTH_BIN", "MONTH"], observed=False
        ).agg(
            temp_mean=("TEMP", "mean"),
            temp_std=("TEMP", "std"),
            temp_p90=("TEMP", lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) >= 5 else np.nan),
            sal_mean=("PSAL", "mean"),
            sal_std=("PSAL", "std"),
            sample_count=("TEMP", "count"),
        ).reset_index()

        # keep enriched columns (MONTH/bins) on final_dataset for historical queries
        self.final_dataset = df
        self.calibrated_clim = self._compute_calibrated_climatology(df)

    def _compute_calibrated_climatology(self, df: pd.DataFrame) -> dict:
        """
        Builds climatology_db for ArgoStatisticalAnomalyEngine FROM REAL DATA
        instead of using its generic hardcoded defaults (which caused every
        real profile to look like an extreme anomaly -- e.g. D20 mu=60m vs
        real Arabian Sea D20 depths of 500-600m).
        """
        surface = df[df["PRES"] <= 30]
        temp_mu = float(surface["TEMP"].mean()) if len(surface) else 27.5
        temp_sigma = float(surface["TEMP"].std()) if len(surface) > 1 else 0.9

        sal_median = float(df["PSAL"].median())
        sal_mad = float((df["PSAL"] - sal_median).abs().median())

        # D20: depth (PRES as proxy) of first crossing below 20C, per cast
        # (grouped by lat/lon/time -- each unique combo ~= one real cast).
        d20_values = []
        for _, cast in df.groupby(["LATITUDE", "LONGITUDE", "JULD"]):
            cast = cast.sort_values("PRES")
            below = cast[cast["TEMP"] <= 20.0]
            if not below.empty:
                d20_values.append(float(below["PRES"].iloc[0]))
        d20_mu = float(np.mean(d20_values)) if d20_values else 60.0
        d20_sigma = float(np.std(d20_values)) if len(d20_values) > 1 else 8.0

        return {
            "temp": {"mu": temp_mu, "sigma": temp_sigma,
                     "p10": temp_mu - 1.28 * temp_sigma, "p90": temp_mu + 1.28 * temp_sigma},
            "sal": {"median": sal_median, "mad": sal_mad if sal_mad > 0 else 0.3},
            "doxy": {"median": 120.0, "mad": 20.0},   # not in this dataset; kept as fallback
            "ph": {"mu": 8.05, "sigma": 0.02},          # not in this dataset; kept as fallback
            "d20": {"mu": d20_mu, "sigma": d20_sigma if d20_sigma > 0 else 8.0},
        }

    def get_nearby_points_categorized(self, lat: float, lon: float, radius_deg: float = 3.0,
                                       max_points: int = 150) -> list:
        """
        Surface (<=30m) readings near (lat, lon), each tagged with a
        Cold/Normal/Warm/Hot category (same z-score bands as the pie chart),
        for the frontend map -- colored dots around the query pin.
        """
        df = self.final_dataset
        nearby = df[(df["LATITUDE"].between(lat - radius_deg, lat + radius_deg)) &
                    (df["LONGITUDE"].between(lon - radius_deg, lon + radius_deg)) &
                    (df["PRES"] <= 30)]
        if nearby.empty:
            return []
        if len(nearby) > max_points:
            nearby = nearby.sample(max_points, random_state=42)

        mu = self.calibrated_clim["temp"]["mu"]
        sigma = self.calibrated_clim["temp"]["sigma"] or 0.9
        z = (nearby["TEMP"] - mu) / sigma

        def categorize(zi):
            if zi < -1:
                return "Cold"
            if zi <= 1:
                return "Normal"
            if zi <= 2:
                return "Warm"
            return "Hot"

        points = []
        for (_, row), zi in zip(nearby.iterrows(), z):
            points.append({
                "lat": round(float(row["LATITUDE"]), 3),
                "lon": round(float(row["LONGITUDE"]), 3),
                "temp": round(float(row["TEMP"]), 2),
                "category": categorize(zi),
            })
        return points
        df = self.final_dataset
        nearby = df[(df["LATITUDE"].between(lat - radius_deg, lat + radius_deg)) &
                    (df["LONGITUDE"].between(lon - radius_deg, lon + radius_deg))]
        return len(nearby)

    def nearby_count(self, lat: float, lon: float, radius_deg: float = 3.0) -> int:
        df = self.final_dataset
        nearby = df[(df["LATITUDE"].between(lat - radius_deg, lat + radius_deg)) &
                    (df["LONGITUDE"].between(lon - radius_deg, lon + radius_deg))]
        return len(nearby)

    def get_real_profile_near(self, lat: float, lon: float, radius_deg: float = 3.0,
                               min_points: int = 5):
        """
        Builds a profile-shaped DataFrame from REAL nearby measurements in
        final_dataset (nearest lat/lon box), for ArgoStatisticalAnomalyEngine.
        Returns None if too few real points are nearby -- caller should then
        fall back to a plain Gemini answer instead of fabricating stats.
        """
        df = self.final_dataset
        nearby = df[(df["LATITUDE"].between(lat - radius_deg, lat + radius_deg)) &
                    (df["LONGITUDE"].between(lon - radius_deg, lon + radius_deg))]
        if len(nearby) < min_points:
            return None
        cols = ["LATITUDE", "LONGITUDE", "PRES", "TEMP", "PSAL", "JULD"]
        profile = nearby[cols].sort_values("PRES").reset_index(drop=True)
        profile["LATITUDE"] = lat
        profile["LONGITUDE"] = lon
        return profile

    def get_real_hist_near(self, lat: float, lon: float, radius_deg: float = 3.0):
        """Daily-ish surface (<=30m) CT proxy timeseries from real data for
        the persistence check. Returns None if not enough days of data."""
        df = self.final_dataset
        nearby = df[(df["LATITUDE"].between(lat - radius_deg, lat + radius_deg)) &
                    (df["LONGITUDE"].between(lon - radius_deg, lon + radius_deg)) &
                    (df["PRES"] <= 30)]
        if nearby.empty:
            return None
        daily = nearby.groupby(nearby["JULD"].dt.date)["TEMP"].mean().reset_index()
        daily.columns = ["DATE", "CT"]
        if len(daily) < 5:
            return None
        return daily


# Single shared instance, created once when the module is first imported
# (FastAPI startup imports this module -> loads once -> reused every request).
store = DataStore()