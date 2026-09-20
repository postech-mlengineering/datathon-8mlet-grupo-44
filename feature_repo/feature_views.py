from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

# A fonte de dados offline apontando para o caminho absoluto no container/projeto
bank_source = FileSource(
    path="../data/bank-additional-full.parquet",
    timestamp_field="event_timestamp",
)

# A entidade principal
client = Entity(name="client_id", join_keys=["client_id"])

# ---------------------------------------------------------
# VERSÃO 1: View original (Com todas as colunas)
# ---------------------------------------------------------
bank_feature_view_v1 = FeatureView(
    name="bank_features",
    entities=[client],
    ttl=timedelta(days=3650),
    tags={"version": "1.0", "status": "production"},
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="job", dtype=String),
        Field(name="marital", dtype=String),
        Field(name="education", dtype=String),
        Field(name="default", dtype=String),
        Field(name="housing", dtype=String),
        Field(name="loan", dtype=String),
        Field(name="contact", dtype=String),
        Field(name="month", dtype=String),
        Field(name="day_of_week", dtype=String),
        Field(name="campaign", dtype=Int64),
        Field(name="pdays", dtype=Int64),
        Field(name="previous", dtype=Int64),
        Field(name="poutcome", dtype=String),
        Field(name="emp.var.rate", dtype=Float64),
        Field(name="cons.price.idx", dtype=Float64),
        Field(name="cons.conf.idx", dtype=Float64),
        Field(name="euribor3m", dtype=Float64),
        Field(name="nr.employed", dtype=Float64),
    ],
    source=bank_source,
)

# ---------------------------------------------------------
# VERSÃO 2: Evolução do modelo (sem month e day_of_week)
# ---------------------------------------------------------
bank_feature_view_v2 = FeatureView(
    name="bank_features_v2",
    entities=[client],
    ttl=timedelta(days=3650),
    tags={"version": "2.0", "status": "development"},
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="job", dtype=String),
        Field(name="marital", dtype=String),
        Field(name="education", dtype=String),
        Field(name="default", dtype=String),
        Field(name="housing", dtype=String),
        Field(name="loan", dtype=String),
        Field(name="contact", dtype=String),
        Field(name="campaign", dtype=Int64),
        Field(name="pdays", dtype=Int64),
        Field(name="previous", dtype=Int64),
        Field(name="poutcome", dtype=String),
        Field(name="emp.var.rate", dtype=Float64),
        Field(name="cons.price.idx", dtype=Float64),
        Field(name="cons.conf.idx", dtype=Float64),
        Field(name="euribor3m", dtype=Float64),
        Field(name="nr.employed", dtype=Float64),
    ],
    source=bank_source,
)