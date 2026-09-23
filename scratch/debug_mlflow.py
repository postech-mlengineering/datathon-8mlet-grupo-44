import os
import sqlite3
import mlflow
from mlflow.tracking.artifact_utils import get_artifact_repository

mlflow.set_tracking_uri("sqlite:///mlflow.db")

print("=== CHECKING MODEL VERSIONS IN DB ===")
conn = sqlite3.connect("mlflow.db")
c = conn.cursor()
for row in c.execute("SELECT name, version, source, storage_location FROM model_versions").fetchall():
    print("Version:", row)
    
print("\n=== TESTING ARTIFACT REPO FOR LATEST ===")
try:
    repo = get_artifact_repository("models:/bank_marketing_model/latest")
    print("Repo artifact_uri:", repo.artifact_uri)
    local_path = repo.download_artifacts("")
    print("Downloaded path:", local_path)
    print("Contents of downloaded path:", os.listdir(local_path))
except Exception as e:
    import traceback
    traceback.print_exc()
