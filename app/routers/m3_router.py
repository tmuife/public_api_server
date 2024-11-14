from fastapi import APIRouter, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKey
from fastapi.params import Security
from app.middleware.auth import get_api_key
from app.services.m3_service import (m3Wrapper,
                                     RequestProcessor,
                                     EmbedRequest,
                                     EmbedResponse,
                                     request_flush_timeout,
                                     max_request, RerankResponse, RerankRequest
                                     )

# Initialize the model and request processor
model = m3Wrapper('BAAI/bge-m3')
processor = RequestProcessor(model, accumulation_timeout=request_flush_timeout, max_request_to_flush=max_request)


router = APIRouter(
    prefix="/secure",
    tags=["SECURE"],
    responses={404: {"message": "Not found"}},
    dependencies=[Security(get_api_key)]
)

@router.post("/embeddings/", response_model=EmbedResponse)
async def get_embeddings(request: EmbedRequest):
    embeddings = await processor.process_request(request, 'embed')
    return EmbedResponse(embeddings=embeddings)


@router.post("/rerank/", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    scores = await processor.process_request(request, 'rerank')
    return RerankResponse(scores=scores)