# Inference Contract — E-commerce Purchase Prediction

## Endpoint intent (future FastAPI)

Predict purchase probability for a single user session.

The model expects one raw session object and returns:
- predicted probability of purchase
- binary decision based on packaged threshold
- model metadata

---

## Input schema (JSON)

A single session object with the same raw columns as the training dataset.

### Required fields

- Administrative: int
- Administrative_Duration: float
- Informational: int
- Informational_Duration: float
- ProductRelated: int
- ProductRelated_Duration: float
- BounceRates: float
- ExitRates: float
- PageValues: float
- SpecialDay: float
- Month: str
- OperatingSystems: int
- Browser: int
- Region: int
- TrafficType: int
- VisitorType: str
- Weekend: bool

### Notes

- `Month` must match values seen during training (unknown values are handled by OneHotEncoder with ignore).
- `VisitorType` must match training categories.
- Numeric values should be non-negative where applicable.
- All fields are required.

---

## Output schema (JSON)

```json
{
  "probability": 0.5467,
  "prediction": 1,
  "threshold": 0.1427,
  "model_version": "v1"
}
```

### Meaning

- probability: predicted probability of purchase (class=1)
- prediction: 1 if probability >= threshold else 0
- threshold: decision threshold packaged with the model bundle
- model_version: version tag of the packaged bundle

---

## Example request

```json
{
  "Administrative": 0,
  "Administrative_Duration": 0.0,
  "Informational": 0,
  "Informational_Duration": 0.0,
  "ProductRelated": 1,
  "ProductRelated_Duration": 10.0,
  "BounceRates": 0.2,
  "ExitRates": 0.3,
  "PageValues": 100.0,
  "SpecialDay": 0.0,
  "Month": "May",
  "OperatingSystems": 2,
  "Browser": 2,
  "Region": 1,
  "TrafficType": 1,
  "VisitorType": "Returning_Visitor",
  "Weekend": false
}
```

---

## Example response

```json
{
  "probability": 0.5467,
  "prediction": 1,
  "threshold": 0.1427,
  "model_version": "v1"
}
```

---

## Internal logic (for engineers)

1. Load model bundle (`model_bundle.joblib`)
2. Extract:
   - sklearn Pipeline
   - threshold
   - model_version
3. Convert input JSON to pandas DataFrame (1 row)
4. Call `predict_proba`
5. Compare probability with threshold
6. Return structured JSON response

---

## Model Bundle Contents

The packaged model bundle contains:

- model: sklearn Pipeline (features → preprocess → logistic regression)
- threshold: business decision threshold
- model_version: version string

---

## Versioning

Current model version: v1  
Threshold packaged: 0.1427  
Training type: Logistic Regression + feature engineering + RandomizedSearchCV