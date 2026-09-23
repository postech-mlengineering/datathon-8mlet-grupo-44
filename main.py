"""Script principal para treinamento, simulação de Multi-Armed Bandit e comparação de versões de features no MLflow."""

import os
import warnings
from datetime import datetime
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Habilita log automático de métricas de sistema (CPU, Memória, Disco)
mlflow.enable_system_metrics_logging()

# =====================================================================
# 1. PREPARAÇÃO DOS DADOS (BASE COMPARTILHADA)
# =====================================================================
print('Carregando e preparando dados base...')
df_raw = pd.read_csv('data/bank-additional-full.csv', sep=';')

df_raw['client_id'] = range(1, len(df_raw) + 1)
df_raw['event_timestamp'] = pd.to_datetime(datetime.now())

os.makedirs('data', exist_ok=True)
features_df = df_raw.drop(columns=['y', 'duration'])
features_df.to_parquet('data/bank-additional-full.parquet')

df_shuffled = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)
y_true = np.where(df_shuffled['y'] == 'yes', 1, 0)


# =====================================================================
# 2. LÓGICA DO BANDIT COM TRACING E OBSERVABILIDADE
# =====================================================================
@mlflow.trace(name="simulate_mab_policy")
def simulate_bandit_and_baseline(
    x_data,
    y_labels,
    raw_data,
    epsilon_start=0.15,
    cost=-0.5,
    reward_success=2.0,
):
    """Simula a política do Multi-Armed Bandit, Baseline e o Oráculo (Regret)."""
    n_samples = x_data.shape[0]
    model = SGDClassifier(
        loss='log_loss', learning_rate='invscaling', eta0=0.1, random_state=42
    )

    bandit_rewards, baseline_rewards, optimal_rewards = [], [], []
    cumulative_bandit, cumulative_baseline, cumulative_optimal = 0, 0, 0
    bandit_offers_made = 0

    warmup = 500
    model.partial_fit(x_data[:warmup], y_labels[:warmup], classes=np.array([0, 1]))

    for t in range(warmup, n_samples):
        context = x_data[t].reshape(1, -1)
        actual_y = y_labels[t]

        # ------------------- 1. Baseline Inteligente -------------------
        row = raw_data.iloc[t]
        baseline_action = 1 if (
            row['poutcome'] == 'success' or row['job'] in ['student', 'retired']
        ) else 0
        r_base = reward_success if (baseline_action == 1 and actual_y == 1) else (cost if baseline_action == 1 else 0)
        cumulative_baseline += r_base
        baseline_rewards.append(cumulative_baseline)

        # ------------------- 2. Oráculo (Modelo Perfeito para Regret) ---
        optimal_action = 1 if actual_y == 1 else 0
        r_optimal = reward_success if (optimal_action == 1 and actual_y == 1) else 0
        cumulative_optimal += r_optimal

        # ------------------- 3. Bandit Policy --------------------------
        current_epsilon = epsilon_start * (1 - (t / n_samples))

        if np.random.rand() < current_epsilon:
            bandit_action = np.random.choice([0, 1])
        else:
            prob_success = model.predict_proba(context)[0][1]
            expected_value_offer = (prob_success * reward_success) + ((1 - prob_success) * cost)
            bandit_action = 1 if expected_value_offer > 0 else 0

        r_bandit = reward_success if (bandit_action == 1 and actual_y == 1) else (cost if bandit_action == 1 else 0)
        cumulative_bandit += r_bandit
        bandit_rewards.append(cumulative_bandit)
        
        if bandit_action == 1:
            bandit_offers_made += 1
            model.partial_fit(context, [actual_y])

        # --- Observabilidade: Logar métricas parciais a cada 1000 steps ---
        if t % 1000 == 0:
            regret = cumulative_optimal - cumulative_bandit
            mlflow.log_metrics({
                'step_bandit_reward': cumulative_bandit,
                'step_baseline_reward': cumulative_baseline,
                'step_regret': regret,
                'step_epsilon': current_epsilon
            }, step=t)

    # Métricas finais da simulação
    final_metrics = {
        "final_bandit_reward": cumulative_bandit,
        "final_baseline_reward": cumulative_baseline,
        "final_regret": cumulative_optimal - cumulative_bandit,
        "offer_rate_percentage": (bandit_offers_made / (n_samples - warmup)) * 100
    }
    
    return model, bandit_rewards, baseline_rewards, final_metrics


# =====================================================================
# 3. PIPELINE PRINCIPAL RASTREADO
# =====================================================================
@mlflow.trace(name="run_feature_experiment")
def run_experiment(view_name, columns, df_shuffled, y_true):
    with mlflow.start_run(run_name=f'bandit_{view_name}') as run:
        print(f"\n--- Treinando e simulando para: {view_name} ---")
        
        # Log Params
        EPSILON_START, COST_REAL, REWARD_REAL = 0.15, -0.5, 2.0
        mlflow.log_params({
            'feature_view_version': view_name,
            'epsilon_start': EPSILON_START,
            'cost_of_failure': COST_REAL,
            'reward_of_success': REWARD_REAL,
            'total_features': len(columns),
            'features_list': str(columns)
        })

        # Processamento
        x_raw = df_shuffled[columns]
        numeric_features = x_raw.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = x_raw.select_dtypes(include=['object', 'category']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ]
        )
        x_processed = preprocessor.fit_transform(x_raw)

        # Simulação (Essa função aparecerá como um nó no Trace do MLflow)
        trained_bandit, rew_bandit, rew_baseline, final_metrics = simulate_bandit_and_baseline(
            x_processed, y_true, df_shuffled,
            epsilon_start=EPSILON_START, cost=COST_REAL, reward_success=REWARD_REAL
        )

        # Log de métricas finais
        mlflow.log_metrics(final_metrics)

        # Observabilidade Visual: Salvar gráfico de performance
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rew_bandit, label='MAB Policy', color='blue')
        ax.plot(rew_baseline, label='Baseline Policy', color='red')
        ax.set_title(f'Cumulative Reward over Time - {view_name}')
        ax.set_xlabel('Interactions (Time Steps)')
        ax.set_ylabel('Cumulative Reward')
        ax.legend()
        ax.grid(True)
        
        # Logar o gráfico como artefato no MLflow
        mlflow.log_figure(fig, f"reward_plot_{view_name}.png")
        plt.close(fig)

        # Log do Modelo
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', trained_bandit),
        ])
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            name='model_pipeline',
            serialization_format='cloudpickle',
            pyfunc_predict_fn='predict_proba',
            registered_model_name='bank_marketing_model'
        )

        print(f'[{view_name}] Lucro: {final_metrics["final_bandit_reward"]} | Regret: {final_metrics["final_regret"]}')
        print(f'Run ID: {run.info.run_id}')


# =====================================================================
# 4. EXECUÇÃO COMPARATIVA
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

for view_name, columns in FEATURE_EXPERIMENTS.items():
    run_experiment(view_name, columns, df_shuffled, y_true)

print('\n--- TODAS AS SIMULAÇÕES CONCLUÍDAS COM SUCESSO ---')