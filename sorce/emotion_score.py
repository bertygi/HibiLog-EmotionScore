# app.py
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# emotion_score.py は同じフォルダに置いてください。
# import 時にモデルがメモリに一度だけロードされるため、推論が高速になります。
from emotion_score import get_combined_score

app = FastAPI(title="Emotion Score API", version="1.0.0")

# CORS: 外部の PC やアプリから直接呼び出す場合、許可ドメインを指定してください。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 本番環境では ["https://your-frontend.example"] など、特定ドメインのみ許可することを推奨
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

class EmotionRequest(BaseModel):
    emoji: str = Field(..., description="絵文字（例：🙂, ❤️）")
    sample: str = Field("", description="分析するテキスト（空でも可）")

class EmotionResponse(BaseModel):
    combined_score_100: float = Field(..., ge=0.0, le=100.0, description="[0〜100] のパーセンテージ")
    # 詳細なデバッグ情報も返したい場合は以下のコメントアウトを解除
    # detail: dict

@app.post("/emotion", response_model=EmotionResponse, summary="絵文字＋テキスト → 感情スコア（％）")
def emotion_endpoint(payload: EmotionRequest):
    # ====== バリデーション ======
    if not payload.emoji.strip():
        raise HTTPException(status_code=400, detail="emoji が空です。")

    # sample（テキスト）は空でも許容（emotion_score 側で w2=0処理）

    try:
        result = get_combined_score(payload.sample, payload.emoji)
        score = result.get("combined_score_100")

        if score is None:
            raise ValueError("combined_score_100 が計算されていません。")

        return EmotionResponse(combined_score_100=score)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"サーバーエラー: {e}")
