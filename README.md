# Datathon Fase 5 - Pós Tech ML Engineering (FIAP)

Repositório do projeto para o datathon da fase 5 da pós-graduação em Machine Learning Engineering da FIAP.

A ideia do projeto é ajudar um banco a decidir para quais clientes vale a pena oferecer um produto numa campanha de marketing. Em vez de ligar para todo mundo ou seguir regras fixas, usamos um modelo que aprende conforme os clientes vão chegando e só recomenda a oferta quando o retorno esperado compensa.

Para isso montamos um Multi-Armed Bandit contextual (epsilon-greedy) com um `SGDClassifier` treinado de forma incremental. Consideramos que cada cliente que aceita a oferta gera +2,0 de ganho, cada recusa custa -0,5 (ligação, tempo do operador, desgaste com o cliente) e não ofertar não gera nem ganho nem custo. O modelo calcula a probabilidade de aceite e só manda "Ofertar" quando o valor esperado é positivo. Comparamos o resultado com um baseline de regras simples (ofertar para quem já converteu antes, estudantes e aposentados) e com um oráculo, que usamos para calcular o regret.

## O que usamos

- **Python 3.11 + Poetry** para o ambiente e as dependências
- **Polars e Pandas** para carregar e preparar os dados
- **scikit-learn** para o pipeline de pré-processamento (`StandardScaler` + `OneHotEncoder`) e o `SGDClassifier`
- **MLflow** para registrar experimentos, métricas, traces e métricas de sistema, para o registry do modelo (alias `champion`) e para servir o modelo
- **Feast** como feature store (offline em parquet, online em SQLite), com duas versões de feature view
- **FastAPI** para a API de inferência, com autenticação OAuth2 simples
- **React + Vite + Tailwind** no frontend do "Motor de Campanha"
- **Docker Compose** para subir a API, o serving do modelo e a UI do MLflow

A base usada é a Bank Marketing (Moro et al., 2014). A coluna `duration` foi removida porque só existe depois que a ligação termina, então usar essa coluna seria vazamento de dados.

## Estrutura

```
.
├── main.py                 # treino/simulação do bandit e registro no MLflow
├── data/                   # base original (csv) e parquet usado pelo Feast
├── feature_repo/           # configuração do Feast e feature views (v1 e v2)
├── backend/api/            # API FastAPI + Dockerfile
├── frontend/               # interface web em React
├── notebooks/eda.ipynb     # análise exploratória e primeiros testes
├── docs/                   # enunciado do datathon
├── Dockerfile.mlflow       # imagem que serve o modelo pelo MLflow
└── docker-compose.yml
```

Sobre as feature views: a `bank_features` (v1) tem todas as colunas, e a `bank_features_v2` tira `month` e `day_of_week`. O `main.py` treina as duas versões e a API usa a v2 por padrão.

## Como executar

### Pré-requisitos

- Python 3.11
- Poetry
- Docker e Docker Compose
- Node 18+ (para o frontend)

### 1. Instalar as dependências

```bash
poetry install
```

### 2. Treinar o modelo

```bash
poetry run python main.py
```

Esse script:
- lê o `data/bank-additional-full.csv`;
- gera o `data/bank-additional-full.parquet` (com `client_id` e `event_timestamp`), que o Feast usa;
- roda a simulação para as duas versões de features e registra tudo no MLflow (`mlflow.db` e pasta `mlruns/`);
- registra o modelo como `bank_marketing_model` e aponta o alias `champion` para a última versão.

### 3. Preparar o Feast

```bash
cd feature_repo
poetry run feast apply
poetry run feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
cd ..
```

O `apply` registra as feature views e o `materialize` carrega os dados no online store (`data/online_store.db`).

### 4. Subir os serviços

```bash
docker compose up --build
```

Esse comando sobe três containers:

| Serviço        | Porta | O que faz                                         |
|----------------|-------|---------------------------------------------------|
| `fastapi_app`  | 8000  | API de inferência                                 |
| `mlflow_serve` | 5002  | serve o `models:/bank_marketing_model@champion`   |
| `mlflow_ui`    | 5000  | interface do MLflow para ver runs e métricas      |

Rode os passos 2 e 3 antes do build, porque a API lê o `feature_repo` e o `data` que são copiados para dentro da imagem.

### 5. Rodar o frontend

```bash
cd frontend
npm install
npm run dev
```

O Vite está configurado na porta 80. **No Linux**, portas abaixo de 1024 só podem ser abertas pelo root, então o `npm run dev` falha com:

```
Error: listen EACCES: permission denied 0.0.0.0:80
```

Nesse caso, suba em outra porta:

```bash
npm run dev -- --port 5173
```

Não é recomendado usar `sudo npm run dev`: o processo inteiro rodaria como root e os arquivos gerados ficariam com dono root.

O login é `admin` / `password123`.

### Endereços

- Frontend: http://localhost (ou http://localhost:5173)
- Documentação da API: http://localhost:8000/docs
- MLflow: http://localhost:5000

## Testando a API direto

Primeiro pegue o token:

```bash
curl -X POST http://localhost:8000/token \
  -d "username=admin&password=password123"
```

Previsão para um cliente específico:

```bash
curl -X POST "http://localhost:8000/predict?client_id=10" \
  -H "Authorization: Bearer admin"
```

Lista dos clientes recomendados para a campanha, ordenados pelo valor esperado (`top_n: 0` retorna todos):

```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Authorization: Bearer admin" \
  -H "Content-Type: application/json" \
  -d '{"top_n": 20}'
```

Cada chamada também gera um run no experimento `bank_marketing_bandit` do MLflow, com a latência, a probabilidade, o valor esperado e a decisão.

## Observações

- Para trocar a feature view usada pela API, defina `FEATURE_VIEW_NAME` (`bank_features` ou `bank_features_v2`) no serviço `fastapi_app`.
- Os arquivos `.db` (MLflow e Feast) estão no `.gitignore`. Por isso, num clone novo, é preciso rodar os passos 2 e 3.
- A autenticação é só para demonstração: o usuário está fixo no código e o token é o próprio nome de usuário.

## Referência da base

S. Moro, P. Cortez e P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, 2014. http://dx.doi.org/10.1016/j.dss.2014.03.001
