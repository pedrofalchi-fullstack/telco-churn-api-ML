"""
main.py
=======
API FastAPI para servir o modelo de previsão de churn de clientes
de telecomunicações (Telco Customer Churn).

Endpoints:
    GET  /         — informações do modelo (nome, classes, features esperadas)
    POST /predict  — recebe features de um cliente, retorna predição + probabilidades

Como rodar localmente:
    pip install -r requirements.txt
    uvicorn main:app --reload

Após subir: http://localhost:8000/docs para a documentação interativa (Swagger).
"""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ============================================================
# Inicialização da aplicação
# ============================================================
app = FastAPI(
    title='Telco Churn Prediction API',
    description='API REST para previsão de cancelamento (churn) de clientes '
                'de telecomunicações usando Regressão Logística.',
    version='1.0.0',
)

# CORS aberto pra permitir consumo por qualquer cliente web
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ============================================================
# Carregamento do modelo (UMA VEZ no startup)
# ============================================================
MODEL_PATH = Path(__file__).parent / 'modelo_final.joblib'

try:
    modelo = joblib.load(MODEL_PATH)
    FEATURES_ESPERADAS = list(modelo.feature_names_in_)
    CLASSES_RAW = modelo.classes_.tolist()  # [0, 1]
    MAPA_CLASSES = {0: 'No Churn', 1: 'Churn'}
    MODELO_OK = True
except Exception as e:
    modelo = None
    FEATURES_ESPERADAS = []
    CLASSES_RAW = []
    MAPA_CLASSES = {}
    MODELO_OK = False
    print(f'ERRO ao carregar modelo: {e}')


# ============================================================
# Schema de entrada — Pydantic valida tipo de CADA feature
# ============================================================
class Cliente(BaseModel):
    """Schema de um cliente para o qual queremos prever churn."""

    # ---- Numéricas (3) ----
    tenure: int = Field(
        ..., ge=0, le=100,
        description='Quantidade de meses como cliente da operadora.'
    )
    MonthlyCharges: float = Field(
        ..., ge=0,
        description='Valor mensal pago pelo cliente (em dólares).'
    )
    TotalCharges: float = Field(
        ..., ge=0,
        description='Valor total pago durante todo o relacionamento.'
    )

    # ---- Categóricas binárias ----
    gender: Literal['Female', 'Male']
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description='1 se cliente é idoso (65+), 0 caso contrário.'
    )
    Partner: Literal['No', 'Yes'] = Field(..., description='Tem parceiro/cônjuge.')
    Dependents: Literal['No', 'Yes'] = Field(..., description='Tem dependentes.')
    PhoneService: Literal['No', 'Yes']
    PaperlessBilling: Literal['No', 'Yes']

    # ---- Categóricas multi-valor ----
    MultipleLines: Literal['No', 'No phone service', 'Yes']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: Literal['No', 'No internet service', 'Yes']
    OnlineBackup: Literal['No', 'No internet service', 'Yes']
    DeviceProtection: Literal['No', 'No internet service', 'Yes']
    TechSupport: Literal['No', 'No internet service', 'Yes']
    StreamingTV: Literal['No', 'No internet service', 'Yes']
    StreamingMovies: Literal['No', 'No internet service', 'Yes']
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    PaymentMethod: Literal[
        'Bank transfer (automatic)',
        'Credit card (automatic)',
        'Electronic check',
        'Mailed check',
    ]

    # Exemplo que aparece no /docs automaticamente
    model_config = {
        'json_schema_extra': {
            'example': {
                'gender': 'Female',
                'SeniorCitizen': 0,
                'Partner': 'Yes',
                'Dependents': 'No',
                'tenure': 12,
                'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'InternetService': 'Fiber optic',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'No',
                'DeviceProtection': 'No',
                'TechSupport': 'No',
                'StreamingTV': 'Yes',
                'StreamingMovies': 'Yes',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 95.5,
                'TotalCharges': 1145.0,
            }
        }
    }


# ============================================================
# Schema de saída — também tipado, vira documentação no /docs
# ============================================================
class Probabilidades(BaseModel):
    no_churn: float = Field(..., alias='No Churn')
    churn: float = Field(..., alias='Churn')

    model_config = {'populate_by_name': True}


class Resposta(BaseModel):
    prediction: int = Field(..., description='0 = No Churn, 1 = Churn')
    label: str = Field(..., description='Rótulo legível: "No Churn" ou "Churn"')
    probabilities: dict = Field(..., description='Probabilidade por classe.')
    risco: str = Field(..., description='Nível de risco: baixo, moderado ou alto.')


# ============================================================
# Tratamento de erros — sempre JSON, status correto
# ============================================================
@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError):
    """Pydantic falhou na validação → 400 (não 500)."""
    return JSONResponse(
        status_code=400,
        content={
            'error': 'Entrada inválida',
            'detail': exc.errors(),
        }
    )


@app.exception_handler(Exception)
async def generic_error_handler(_request: Request, exc: Exception):
    """Qualquer outra exceção não tratada → 500 com JSON, não com HTML."""
    return JSONResponse(
        status_code=500,
        content={
            'error': 'Erro interno do servidor',
            'type': type(exc).__name__,
            'detail': str(exc),
        }
    )


# ============================================================
# Endpoints
# ============================================================
@app.get('/')
def info():
    """
    Retorna informações do modelo: nome, classes possíveis e features esperadas.
    Esse endpoint serve como auto-documentação do contrato da API.
    """
    if not MODELO_OK:
        raise HTTPException(status_code=503, detail='Modelo não carregado')

    return {
        'modelo': 'Telco Customer Churn — Regressão Logística',
        'descricao': 'Prevê se um cliente de operadora de telecomunicações '
                     'irá cancelar (churn) o serviço no próximo ciclo.',
        'versao': '1.0.0',
        'classes': {
            'codigos': CLASSES_RAW,                 # [0, 1]
            'rotulos': list(MAPA_CLASSES.values()), # ['No Churn', 'Churn']
            'mapeamento': MAPA_CLASSES,             # {0: 'No Churn', 1: 'Churn'}
        },
        'features_esperadas': FEATURES_ESPERADAS,
        'total_features': len(FEATURES_ESPERADAS),
        'tipo_problema': 'Classificação binária supervisionada',
        'pipeline': [
            'ColumnTransformer (StandardScaler em 3 numéricas + '
            'OneHotEncoder em 16 categóricas)',
            'LogisticRegression',
        ],
        'exemplo_request': Cliente.model_config['json_schema_extra']['example'],
    }


@app.post('/predict', response_model=Resposta)
def predict(cliente: Cliente):
    """
    Recebe os dados de um cliente e retorna a predição de churn
    com as probabilidades de cada classe.
    """
    if not MODELO_OK:
        raise HTTPException(status_code=503, detail='Modelo não carregado')

    # Converte Pydantic → DataFrame de 1 linha na ordem que o pipeline espera
    dados = cliente.model_dump()
    df = pd.DataFrame([dados])[FEATURES_ESPERADAS]

    # Predição
    pred = int(modelo.predict(df)[0])
    proba = modelo.predict_proba(df)[0]

    # Monta resposta amigável
    p_churn = float(proba[1])
    if p_churn >= 0.75:
        risco = 'alto'
    elif p_churn >= 0.4:
        risco = 'moderado'
    else:
        risco = 'baixo'

    return Resposta(
        prediction=pred,
        label=MAPA_CLASSES[pred],
        probabilities={
            'No Churn': round(float(proba[0]), 4),
            'Churn': round(float(proba[1]), 4),
        },
        risco=risco,
    )
