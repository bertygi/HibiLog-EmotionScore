# app.py
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# emotion_score.py는 같은 폴더에 둡니다.
# import 시 모델이 메모리에 1회 로드되어 추론이 빠릅니다.
from emotion_score import get_combined_score

app = FastAPI(title="Emotion Score API", version="1.0.0")

# CORS: 외부 PC에서 직접 호출한다면 허용 도메인을 지정하세요.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 운영 시 ["https://your-frontend.example"] 등으로 제한 권장
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

class EmotionRequest(BaseModel):
    emoji: str = Field(..., description="이모지 문자 (예: 🙂, ❤️)")
    sample: str = Field(..., description="분석할 텍스트")

class EmotionResponse(BaseModel):
    combined_score_100: float = Field(..., ge=0.0, le=100.0, description="[0,100] 백분율")
    # 필요 시 상세 디버그 값도 함께 반환하고 싶다면 아래 주석을 해제
    # detail: dict

@app.post("/emotion", response_model=EmotionResponse, summary="이모지+텍스트 → 감정점수(%)")
def emotion_endpoint(payload: EmotionRequest):
    # 간단한 유효성 검사
    if not payload.emoji.strip():
        raise HTTPException(status_code=400, detail="emoji가 비어있습니다.")
    if not payload.sample.strip():
        raise HTTPException(status_code=400, detail="sample이 비어있습니다.")

    try:
        result = get_combined_score(payload.sample, payload.emoji)
        score = result.get("combined_score_100")
        if score is None:
            raise ValueError("combined_score_100이 계산되지 않았습니다.")
        return EmotionResponse(combined_score_100=score)
        # 상세 반환을 원하면:
        # return {"combined_score_100": score, "detail": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {e}")
