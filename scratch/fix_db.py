import os
import sqlite3

db_path = "mlflow.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

tables_cols = [
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("model_versions", "storage_location"),
    ("trace_tags", "value")
]

print("=== BEFORE UPDATE ===")
for table, col in tables_cols:
    print(f"--- {table}.{col} ---")
    for r in c.execute(f"SELECT {col} FROM {table}").fetchall():
        print("  ", r[0])

for table, col in tables_cols:
    c.execute(f"""
        UPDATE {table}
        SET {col} = REPLACE({col}, '/home/platero/postech-ml-engineering-fase-5-datathon', '/app')
    """)

conn.commit()

print("\n=== AFTER UPDATE ===")
for table, col in tables_cols:
    print(f"--- {table}.{col} ---")
    for r in c.execute(f"SELECT {col} FROM {table}").fetchall():
        print("  ", r[0])

conn.close()

# Also update MLmodel files to set artifact_path: ""
for root, dirs, files in os.walk("mlruns"):
    for f in files:
        if f == "MLmodel":
            filepath = os.path.join(root, f)
            with open(filepath, "r") as file:
                content = file.read()
            content = content.replace("/home/platero/postech-ml-engineering-fase-5-datathon", "/app")
            lines = content.splitlines(keepends=True)
            if len(lines) > 0 and "artifact_path:" in lines[0]:
                lines[0] = 'artifact_path: ""\n'
            with open(filepath, "w") as file:
                file.writelines(lines)

print("DB and MLmodel files fix complete.")
