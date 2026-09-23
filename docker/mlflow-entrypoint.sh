#!/bin/sh
set -eu

cd /app

: "${MLFLOW_TRACKING_URI:=sqlite:////app/mlflow.db}"
export MLFLOW_TRACKING_URI

if python - <<'PY'
import mlflow

model_uri = "models:/bank_marketing_model/latest"

try:
    mlflow.pyfunc.load_model(model_uri)
    print(f"Modelo registrado encontrado: {model_uri}")
except Exception as exc:
    print(f"Modelo registrado indisponivel ({exc}). Treinando e registrando novamente...")
    raise SystemExit(1)
PY
then
    :
else
    python main.py
fi

exec mlflow models serve \
    -m models:/bank_marketing_model/latest \
    --host 0.0.0.0 \
    --port 5002 \
    --env-manager local
