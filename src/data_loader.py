import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CREATURE_COLORS = {
    "Ghoul": "#10b981",    # Emerald
    "Goblin": "#f59e0b",   # Amber / Orange
    "Ghost": "#38bdf8"     # Cyan / Ethereal
}

COLOR_MAP = {
    "clear": "#94a3b8",
    "green": "#22c55e",
    "black": "#334155",
    "white": "#f8fafc",
    "blue": "#3b82f6",
    "blood": "#ef4444"
}

def load_monster_data(csv_path="data/monsters.csv"):
    df = pd.read_csv(csv_path)
    return df

def train_voting_classifier(df):
    """Trains an Ensemble Voting Classifier combining Random Forest, Extra Trees, and Gradient Boosting."""
    features = ["bone_length", "rotting_flesh", "hair_length", "has_soul", "color"]
    X = df[features]
    y = df["type"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["color"])
        ],
        remainder="passthrough"
    )
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    et = ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42)
    
    voting_ensemble = VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("gb", gb)],
        voting="soft"
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", voting_ensemble)
    ])
    
    pipeline.fit(X, y)
    
    return pipeline, {"rf": rf, "et": et, "gb": gb}
