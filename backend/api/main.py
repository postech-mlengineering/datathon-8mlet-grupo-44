"""Aplicação FastAPI para inferência em tempo real integrada ao Feast e MLflow com Observabilidade."""

import os
import time

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from feast import FeatureStore
from mlflow.tracking import MlflowClient
import polars as pl
from pydantic import BaseModel
import requests

# Configuração da URL de tracking do MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_serve:5002")

app = FastAPI(title="Bank Marketing API MLOps")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = FeatureStore(repo_path="/app/feature_repo")

# --- SISTEMA DE AUTENTICAÇÃO ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
FAKE_USERS_DB = {"admin": {"username": "admin", "password": "password123"}}


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Autentica o usuário e retorna o token de acesso."""
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or form_data.password != user["password"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Usuário ou senha incorretos",
        )
    return {"access_token": user["username"], "token_type": "bearer"}


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Valida o token de acesso atual."""
    if token not in FAKE_USERS_DB:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return token


# --- CONFIGURAÇÃO DINÂMICA DA FEATURE VIEW ATIVA ---
ACTIVE_FEATURE_VIEW = os.getenv("FEATURE_VIEW_NAME", "bank_features_v2")

# Mapeamento estrito baseado apenas na versão ativa configurada no ambiente
if ACTIVE_FEATURE_VIEW == "bank_features_v2":
    MODEL_COLUMNS = [
        "age", "job", "marital", "education", "default", "housing", "loan",
        "contact", "campaign", "pdays", "previous", "poutcome",
        "emp.var.rate", "cons.price.idx", "cons.conf.idx",
        "euribor3m", "nr.employed",
    ]
elif ACTIVE_FEATURE_VIEW == "bank_features":
    MODEL_COLUMNS = [
        "age", "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "campaign", "pdays", "previous",
        "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
        "euribor3m", "nr.employed",
    ]
else:
    raise ValueError(
        f"Feature View '{ACTIVE_FEATURE_VIEW}' não suportada pela API."
    )


# =====================================================================
# ROTA 1: PREVISÃO INDIVIDUAL
# =====================================================================
@app.post("/predict")
def predict_oferta(client_id: int, token: str = Depends(get_current_user)):
    """Busca features no Feast, realiza predição no MLflow e loga observabilidade de forma thread-safe."""
    start_time = time.time()
    
    # Cliente do MLflow isolado para evitar vazamento de contexto em requisições simultâneas
    mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = mlflow_client.get_experiment_by_name("bank_marketing_bandit")
    exp_id = experiment.experiment_id if experiment else mlflow_client.create_experiment("bank_marketing_bandit")

    try:
        feature_vector = store.get_online_features(
            features=[f"{ACTIVE_FEATURE_VIEW}:{col}" for col in MODEL_COLUMNS],
            entity_rows=[{"client_id": client_id}],
        ).to_dict()

        if feature_vector["age"][0] is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Cliente não encontrado no Feature Store",
            )

        data_row = [feature_vector[col][0] for col in MODEL_COLUMNS]
        mlflow_payload = {
            "dataframe_split": {
                "columns": MODEL_COLUMNS,
                "data": [data_row],
            }
        }

        mlflow_url = os.getenv("MLFLOW_URL", "http://mlflow_serve:5002/invocations")
        response = requests.post(mlflow_url, json=mlflow_payload, timeout=10)

        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erro no serviço de modelo ML",
            )

        prob_success = response.json()["predictions"][0][1]
        expected_value = (prob_success * 2.0) + ((1 - prob_success) * -0.5)
        decision = "Ofertar" if expected_value > 0 else "Não ofertar"

        latency = time.time() - start_time

        try:
            run = mlflow_client.create_run(exp_id, run_name=f"inference-client-{client_id}")
            mlflow_client.log_param(run.info.run_id, "client_id", client_id)
            mlflow_client.log_param(run.info.run_id, "feature_view", ACTIVE_FEATURE_VIEW)
            mlflow_client.log_metric(run.info.run_id, "inference_latency_seconds", latency)
            mlflow_client.log_metric(run.info.run_id, "probabilidade_aceitacao", prob_success)
            mlflow_client.log_metric(run.info.run_id, "valor_esperado", expected_value)
            mlflow_client.set_tag(run.info.run_id, "decision", decision)
            mlflow_client.set_tag(run.info.run_id, "model_status", "success")
            mlflow_client.set_tag(run.info.run_id, "endpoint", "/predict")
            mlflow_client.set_terminated(run.info.run_id)
        except Exception as obs_err:
            print(f"Aviso: Falha ao registrar métricas no MLflow: {obs_err}")

        return {
            "client_id": client_id,
            "versao_feature_view": ACTIVE_FEATURE_VIEW,
            "probabilidade_aceitacao": round(prob_success, 4),
            "valor_esperado": round(expected_value, 2),
            "decisao": decision,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        try:
            run = mlflow_client.create_run(exp_id, run_name=f"inference-error-client-{client_id}")
            mlflow_client.log_param(run.info.run_id, "client_id", client_id)
            mlflow_client.log_param(run.info.run_id, "feature_view", ACTIVE_FEATURE_VIEW)
            mlflow_client.set_tag(run.info.run_id, "model_status", "error")
            mlflow_client.set_tag(run.info.run_id, "error_message", str(e))
            mlflow_client.set_terminated(run.info.run_id, status="FAILED")
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# =====================================================================
# ROTA 2: RECOMENDAÇÕES EM LOTE
# =====================================================================
class CampaignRequest(BaseModel):
    top_n: int = 0


@app.post("/recommendations")
def get_recommendations(
    request: CampaignRequest, token: str = Depends(get_current_user)
):
    """Analisa todos os clientes da base, ordena e retorna os Top N aprovados."""
    start_time = time.time()
    
    # Cliente do MLflow isolado
    mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = mlflow_client.get_experiment_by_name("bank_marketing_bandit")
    exp_id = experiment.experiment_id if experiment else mlflow_client.create_experiment("bank_marketing_bandit")

    try:
        try:
            # Polars para extração em lote otimizada
            df_base = pl.read_parquet("/app/data/bank-additional-full.parquet")
            client_ids = df_base["client_id"].to_list()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Base de dados não encontrada no servidor da API.",
            ) from exc

        entity_rows = [{"client_id": cid} for cid in client_ids]
        feature_vector = store.get_online_features(
            features=[f"{ACTIVE_FEATURE_VIEW}:{col}" for col in MODEL_COLUMNS],
            entity_rows=entity_rows,
        ).to_dict()

        data_rows = []
        valid_clients = []

        for i, cid in enumerate(client_ids):
            if feature_vector["age"][i] is not None:
                row = [feature_vector[col][i] for col in MODEL_COLUMNS]
                data_rows.append(row)
                valid_clients.append(cid)

        mlflow_payload = {
            "dataframe_split": {
                "columns": MODEL_COLUMNS,
                "data": data_rows,
            }
        }

        mlflow_url = os.getenv("MLFLOW_URL", "http://mlflow_serve:5002/invocations")
        response = requests.post(mlflow_url, json=mlflow_payload, timeout=60)

        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erro no serviço de modelo ML",
            )

        predictions = response.json()["predictions"]

        recommended_clients = []
        for cid, pred in zip(valid_clients, predictions):
            prob_success = pred[1]
            expected_value = (prob_success * 2.0) + ((1 - prob_success) * -0.5)

            if expected_value > 0:
                recommended_clients.append({
                    "client_id": cid,
                    "probabilidade_aceitacao": round(prob_success, 4),
                    "valor_esperado": round(expected_value, 2),
                    "decisao": "Ofertar",
                })

        recommended_clients.sort(
            key=lambda x: x["valor_esperado"], reverse=True
        )

        if request.top_n > 0:
            recommended_clients = recommended_clients[: request.top_n]

        latency = time.time() - start_time
        
        try:
            run = mlflow_client.create_run(exp_id, run_name="inference-campaign")
            mlflow_client.log_param(run.info.run_id, "total_analisados", len(valid_clients))
            mlflow_client.log_param(run.info.run_id, "feature_view", ACTIVE_FEATURE_VIEW)
            mlflow_client.log_param(run.info.run_id, "top_n_solicitado", request.top_n)
            mlflow_client.log_metric(run.info.run_id, "total_recomendados", len(recommended_clients))
            mlflow_client.log_metric(run.info.run_id, "inference_latency_seconds", latency)
            mlflow_client.set_terminated(run.info.run_id)
        except Exception as obs_err:
            print(f"Aviso: Falha ao registrar métricas da campanha no MLflow: {obs_err}")

        return {
            "resumo": {
                "versao_feature_view": ACTIVE_FEATURE_VIEW,
                "total_processados": len(valid_clients),
                "total_retornados": len(recommended_clients),
            },
            "recomendacoes": recommended_clients,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        try:
            run = mlflow_client.create_run(exp_id, run_name="inference-error-campaign")
            mlflow_client.set_tag(run.info.run_id, "model_status", "error")
            mlflow_client.set_tag(run.info.run_id, "error_message", str(e))
            mlflow_client.set_terminated(run.info.run_id, status="FAILED")
        except Exception:
            pass
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )