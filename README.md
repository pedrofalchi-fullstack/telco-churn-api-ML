# Telco Churn — API de Predição (FastAPI)

API REST para previsão de cancelamento (churn) de clientes de operadora de
telecomunicações, baseada em um pipeline scikit-learn (`StandardScaler` +
`OneHotEncoder` + `LogisticRegression`) treinado sobre o dataset
**IBM Telco Customer Churn**.

## Stack

| Componente | Versão |
|---|---|
| FastAPI | ≥ 0.110 |
| Uvicorn | ≥ 0.27 (com extras `standard`) |
| Pydantic | ≥ 2.0 |
| scikit-learn | ≥ 1.5, < 2.0 |
| pandas | ≥ 2.0 |
| joblib | ≥ 1.3 |

## Estrutura do projeto

```
.
├── main.py                 # Aplicação FastAPI
├── modelo_final.joblib     # Pipeline treinado (não versionar se grande)
├── requirements.txt        # Dependências
└── README.md
```

## Como rodar localmente

```bash
# 1. Criar e ativar virtualenv (opcional mas recomendado)
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Subir o servidor
uvicorn main:app --reload
```

A API sobe em `http://localhost:8000`.

## Endpoints

### `GET /`

Retorna metadados do modelo: nome, classes, features esperadas,
descrição do pipeline e um exemplo de request válido.

```bash
curl http://localhost:8000/
```

### `POST /predict`

Recebe um JSON com os 19 atributos de um cliente e devolve a predição
de churn com as probabilidades.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 1145.0
  }'
```

Resposta esperada:
```json
{
  "prediction": 1,
  "label": "Churn",
  "probabilities": {
    "No Churn": 0.1318,
    "Churn": 0.8682
  },
  "risco": "alto"
}
```

## Documentação interativa

Com o servidor rodando, abra no navegador:

- **Swagger UI** → http://localhost:8000/docs
- **ReDoc** → http://localhost:8000/redoc

Nessas páginas dá pra testar a API direto, ver o schema completo do request
e response, e copiar exemplos prontos.

## Validação de entrada (Pydantic)

Todos os 19 campos são tipados via `Literal` (para categóricas) ou tipos
numéricos com restrições (`ge`/`le`). Requisições com:

- Campos faltando
- Tipos errados (ex.: string em campo numérico)
- Valores fora do vocabulário do treino

...retornam **HTTP 400** com JSON detalhando o problema, em vez de quebrar
o servidor com 500.

## Vocabulário aceito por campo

| Campo | Valores válidos |
|---|---|
| `gender` | `Female`, `Male` |
| `SeniorCitizen` | `0`, `1` |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | `No`, `Yes` |
| `MultipleLines` | `No`, `No phone service`, `Yes` |
| `InternetService` | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | `No`, `No internet service`, `Yes` |
| `Contract` | `Month-to-month`, `One year`, `Two year` |
| `PaymentMethod` | `Bank transfer (automatic)`, `Credit card (automatic)`, `Electronic check`, `Mailed check` |
| `tenure` | inteiro 0–100 |
| `MonthlyCharges`, `TotalCharges` | float ≥ 0 |

## Sobre o modelo

O pipeline serializado em `modelo_final.joblib` contém **todo o
pré-processamento embutido**. O cliente da API só precisa enviar os
atributos brutos no formato do dataset original — sem cálculo de features
derivadas, encoding manual nem normalização.

- Algoritmo: `LogisticRegression` (escolhido por priorizar recall sobre a
  classe minoritária, conforme análise no notebook de treino)
- Pré-processamento: `StandardScaler` em 3 numéricas + `OneHotEncoder` em
  16 categóricas, dentro de um `ColumnTransformer`

## Status

API testada localmente com `fastapi.testclient.TestClient`:

- [x] `GET /` retornando schema completo
- [x] `POST /predict` para cliente alto risco → Churn (~87%)
- [x] `POST /predict` para cliente baixo risco → No Churn (~98%)
- [x] Validação de vocabulário rejeita valores inválidos com 400
- [x] Campo obrigatório faltando rejeitado com 400
- [x] Tipo incorreto rejeitado com 400
- [x] Modelo carregado uma única vez no startup
