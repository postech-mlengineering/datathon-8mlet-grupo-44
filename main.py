"""Script principal para treinamento, simulação de Multi-Armed Bandit e comparação de versões de features no MLflow."""

import os
import warnings
from datetime import datetime
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

# =====================================================================
# 0. CONFIGURAÇÃO DE DIRETÓRIOS E MLFLOW
# =====================================================================
os.makedirs('mlruns', exist_ok=True)
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('bank_marketing_bandit')
mlflow.autolog(disable=True)


# =====================================================================
# 1. PREPARAÇÃO DOS DADOS (BASE COMPARTILHADA)
# =====================================================================
print('Carregando e preparando dados base...')
df_raw = pd.read_csv('data/bank-additional-full.csv', sep=';')

df_raw['client_id'] = range(1, len(df_raw) + 1)
df_raw['event_timestamp'] = pd.to_datetime(datetime.now())

# O Feast pode manter a base geral, mas vamos separar as views/features nas execuções
features_df = df_raw.drop(columns=['y', 'duration'])
features_df.to_parquet('data/bank-additional-full.parquet')

df_shuffled = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)
y_true = np.where(df_shuffled['y'] == 'yes', 1, 0)


# =====================================================================
# 2. LÓGICA DO BANDIT
# =====================================================================
def simulate_bandit_and_baseline(
    x_data,
    y_labels,
    raw_data,
    epsilon_start=0.15,
    cost=-0.5,
    reward_success=2.0,
):
    """Simula a política do Multi-Armed Bandit em comparação ao baseline."""
    n_samples = x_data.shape[0]
    model = SGDClassifier(
        loss='log_loss', learning_rate='invscaling', eta0=0.1, random_state=42
    )

    bandit_rewards, baseline_rewards = [], []
    cumulative_bandit_reward, cumulative_baseline_reward = 0, 0

    warmup = 500
    model.partial_fit(x_data[:warmup], y_labels[:warmup], classes=np.array([0, 1]))

    for t in range(warmup, n_samples):
        context = x_data[t].reshape(1, -1)
        actual_y = y_labels[t]

        # Baseline Inteligente
        row = raw_data.iloc[t]
        baseline_action = 1 if (
            row['poutcome'] == 'success' or row['job'] in ['student', 'retired']
        ) else 0

        r_base = (
            reward_success
            if (baseline_action == 1 and actual_y == 1)
            else (cost if baseline_action == 1 else 0)
        )
        cumulative_baseline_reward += r_base
        baseline_rewards.append(cumulative_baseline_reward)

        # Bandit
        current_epsilon = epsilon_start * (1 - (t / n_samples))

        if np.random.rand() < current_epsilon:
            bandit_action = np.random.choice([0, 1])
        else:
            prob_success = model.predict_proba(context)[0][1]
            expected_value_offer = (prob_success * reward_success) + (
                (1 - prob_success) * cost
            )
            bandit_action = 1 if expected_value_offer > 0 else 0

        r_bandit = (
            reward_success
            if (bandit_action == 1 and actual_y == 1)
            else (cost if bandit_action == 1 else 0)
        )
        cumulative_bandit_reward += r_bandit
        bandit_rewards.append(cumulative_bandit_reward)

        if bandit_action == 1:
            model.partial_fit(context, [actual_y])

    return model, bandit_rewards, baseline_rewards


# =====================================================================
# 3. EXECUÇÃO COMPARATIVA DAS DUAS VERSÕES (COM E SEM MESES/DIAS)
# =====================================================================
FEATURE_EXPERIMENTS = {
    "bank_features_v1": [
        'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous',
        'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
        'euribor3m', 'nr.employed',
    ],
    "bank_features_v2": [
        'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'campaign', 'pdays', 'previous', 'poutcome',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
        'euribor3m', 'nr.employed',
    ],
}

EPSILON_START = 0.15
COST_REAL = -0.5
REWARD_REAL = 2.0

for view_name, columns in FEATURE_EXPERIMENTS.items():
    print(f"\n--- Treinando e simulando para: {view_name} ---")
    
    # Prepara subset de colunas para treino
    x_raw = df_shuffled[columns]

    numeric_features = x_raw.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = x_raw.select_dtypes(
        include=['object', 'category']
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )

    print(f'Processando features para {view_name}...')
    x_processed = preprocessor.fit_transform(x_raw)

    with mlflow.start_run(run_name=f'bandit_{view_name}') as run:
        mlflow.log_params({
            'feature_view_version': view_name,
            'epsilon_start': EPSILON_START,
            'cost_of_failure': COST_REAL,
            'reward_of_success': REWARD_REAL,
            'total_features': len(columns),
        })

        trained_bandit, rew_bandit, rew_baseline = simulate_bandit_and_baseline(
            x_processed,
            y_true,
            df_shuffled,
            epsilon_start=EPSILON_START,
            cost=COST_REAL,
            reward_success=REWARD_REAL,
        )

        mlflow.log_metrics({
            'bandit_reward': rew_bandit[-1],
            'baseline_reward': rew_baseline[-1],
        })

        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', trained_bandit),
        ])

        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path='model_pipeline',
            serialization_format='cloudpickle',
            pyfunc_predict_fn='predict_proba',
        )

        print(f'[{view_name}] Concluído! Lucro Bandit: {rew_bandit[-1]} | Run ID: {run.info.run_id}')

print('\n--- TODAS AS SIMULAÇÕES CONCLUÍDAS COM SUCESSO ---')