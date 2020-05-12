import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(MODEL_DIR, "baseline_model_modified.pt")
OUTPUT_DIR = os.path.join(MODEL_DIR, "results")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)