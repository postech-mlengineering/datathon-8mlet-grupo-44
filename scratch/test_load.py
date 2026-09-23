import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

for v in [1, 2, 3, 4]:
    try:
        m = mlflow.pyfunc.load_model(f"models:/bank_marketing_model/{v}")
        print(f"Version {v}: SUCCESS - {m}")
    except Exception as e:
        print(f"Version {v}: FAILED - {e}")

try:
    m = mlflow.pyfunc.load_model("models:/bank_marketing_model/latest")
    print(f"Latest: SUCCESS - {m}")
except Exception as e:
    print(f"Latest: FAILED - {e}")
