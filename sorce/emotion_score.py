# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import math

# ============================
# ハイパーパラメータ
# ============================
MODEL_ID = "../bert-ja-wrime"

ALPHA = 0.5   # w2 = α*c + β
BETA = 0.2
EPS = 1e-6

EMOJI_SCORE = {
    "❤️": +0.9,
    "🙂": +0.3,
    "😢": -0.6,
    "😱": -0.9,
    "😖": -0.5,
    "🔥": +0.5,
    "😡": -0.8,
    "👍": +0.6,
}

# ============================
# 感情ラベルの順序（WRIME基準）
# ============================
EMOTIONS = ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"]

# ============================
# モデルのロード
# ============================
print("モデルを読み込み中...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()

# ============================
# 感情スコア計算関数（LSEベース）
# ============================
@torch.inference_mode()
def get_text_sent_score(text: str, anticipation_weight: float = 0.5):
    """
    WRIMEモデルを使用して文の感情スコア（text_sent_score）を算出（log-sum-expベース）
    - 出力が16の場合（Writer 8 + Reader 8）→ Reader（後半8個）を使用
    - posグループ: joy, trust, anticipation（重みを適用）
    - negグループ: sadness, anger, fear, disgust
    - 代表ロジット L_pos/L_neg = logsumexp(各グループのロジット + log(重み))
    - p_pos = sigmoid(L_pos - L_neg)
    - text_sent_score = 2 * p_pos - 1  （範囲: [-1, 1]）
    - c（信頼度）: 2クラス分布のエントロピーに基づく信頼度（1 - H / log 2）
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    logits = model(**enc).logits.squeeze(0)  # shape: (8,) または (16,)

    # 出力が16の場合は後半8個（Reader）を使用、8の場合はそのまま使用
    if logits.shape[0] == 16:
        logits = logits[8:16]
    elif logits.shape[0] != 8:
        raise RuntimeError(f"出力サイズが想定外です: {logits.shape[0]} （想定は8または16）")

    # 感情ごとのロジット（softmax前の値）をマッピング
    emotion_values = {emo: float(logits[i].item()) for i, emo in enumerate(EMOTIONS)}

    # 各感情のロジットを抽出（テンソルのまま保持）
    joy        = logits[0]
    sadness    = logits[1]
    anticipation = logits[2]
    # surprise  = logits[3]  # 現在はpos/neg計算に使用しない
    anger      = logits[4]
    fear       = logits[5]
    disgust    = logits[6]
    trust      = logits[7]

    # --------- LSEで代表ロジットを計算 ---------
    # anticipationの重み0.5はlog空間でlog(0.5)を加算して反映
    log_w_a = torch.log(torch.tensor(anticipation_weight, device=logits.device))
    pos_stack = torch.stack([joy, trust, anticipation + log_w_a])  # ポジティブグループ
    neg_stack = torch.stack([sadness, anger, fear, disgust])       # ネガティブグループ

    L_pos = torch.logsumexp(pos_stack, dim=0)
    L_neg = torch.logsumexp(neg_stack, dim=0)

    # 2クラスsoftmax確率（sigmoid(delta)）
    delta = L_pos - L_neg
    p_pos = torch.sigmoid(delta)         # ポジティブ確率
    p_neg = 1.0 - p_pos

    # 連続スコア [-1, 1]
    text_sent_score = float(2.0 * p_pos.item() - 1.0)

    # 信頼度c: 2クラス分布のエントロピーに基づく（0~1）
    # H = -Σ p log p, c = 1 - H / log(2)
    p_pos_clamped = torch.clamp(p_pos, EPS, 1.0 - EPS)  # 数値安定化
    p_neg_clamped = 1.0 - p_pos_clamped
    H = -(p_pos_clamped * torch.log(p_pos_clamped) + p_neg_clamped * torch.log(p_neg_clamped))
    c = float(1.0 - (H.item() / math.log(2.0)))

    return text_sent_score, c, emotion_values

# ============================
# combined_score 計算関数
# ============================
def get_combined_score(text: str, emoji: str):
    text_sent_score, c, emotions = get_text_sent_score(text)

    # w2 = α*c + β （0~1でクリッピング）
    w2 = float(max(0.0, min(1.0, ALPHA * c + BETA)))
    w1 = 1.0 - w2

    emoji_score = EMOJI_SCORE.get(emoji, 0.0)
    combined = w1 * emoji_score + w2 * text_sent_score

    # --- 0~100スケールにリスケーリング ---
    combined_rescaled = ((combined + 1.0) / 2.0) * 100.0
    combined_rescaled = float(max(0.0, min(100.0, combined_rescaled)))  # 安全なクリッピング

    return {
        "text": text,
        "emoji": emoji,
        "emoji_score": round(emoji_score, 3),
        "text_sent_score": round(text_sent_score, 3),
        "confidence(c)": round(c, 3),
        "w1": round(w1, 3),
        "w2": round(w2, 3),
        "combined_score": round(combined, 3),  # [-1, 1]
        "combined_score_100": round(combined_rescaled, 2),  # [0, 100]
        "emotion_values": {k: round(v, 3) for k, v in emotions.items()},
    }


# ============================
# 実行例
# ============================
if __name__ == "__main__":
    sample = "今日は友達と会えてとても嬉しい！"
    emoji = "🙂"
    result = get_combined_score(sample, emoji)

    print("\n--- 感情スコア結果 ---")
    for k, v in result.items():
        print(f"{k}: {v}")
