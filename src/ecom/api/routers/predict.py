from fastapi import HTTPException, APIRouter

from ecom.api.schemas import PredictRequest, PredictResponse
from ecom.inference import predict_one, load_bundle

router = APIRouter()

_bundle = None

def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_bundle()
    return _bundle

@router.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        bundle = get_bundle()
        result = predict_one(req.model_dump(), bundle)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))