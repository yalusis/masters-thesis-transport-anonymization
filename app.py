import json
from pathlib import Path
from flask import Flask, jsonify, render_template
import pandas as pd

app = Flask(__name__)

OUTPUT_DIR = Path("./output")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    result = {}

    report_path = OUTPUT_DIR / "anonymization_report.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            result["report"] = json.load(f)

    csv_path = OUTPUT_DIR / "anonymized_data.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)

        if "sa_behavior" in df.columns:
            result["behavior_dist"] = df["sa_behavior"].value_counts().to_dict()

        if "road_type" in df.columns:
            result["road_dist"] = df["road_type"].value_counts().to_dict()

        if "qi_speed_cat" in df.columns:
            result["speed_dist"] = df["qi_speed_cat"].value_counts().to_dict()

        qi_cols = ["qi_geohash", "qi_time_bucket", "qi_speed_cat"]
        if all(c in df.columns for c in qi_cols):
            sizes = df.groupby(qi_cols).size().sort_values(ascending=False)
            result["eq_sizes"] = sizes.head(20).tolist()

        if "score_total" in df.columns and "sa_behavior" in df.columns:
            scores = df.groupby("sa_behavior")["score_total"].mean().round(1).to_dict()
            result["avg_scores"] = scores

        if "qi_geohash" in df.columns:
            result["geohash_dist"] = df["qi_geohash"].value_counts().head(10).to_dict()

    trips_path = OUTPUT_DIR / "trips_summary.csv"
    if trips_path.exists():
        trips_df = pd.read_csv(trips_path)
        result["trips"] = trips_df.fillna("—").to_dict(orient="records")

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)