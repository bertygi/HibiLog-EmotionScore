# 🧠 Emotion Score Program
絵文字とテキストを組み合わせた感情分析プログラム

---

## 📘 概要
本プログラムは、**Hugging Face** の学習済み日本語感情分析モデル  
[`patrickramos/bert-base-japanese-v2-wrime-fine-tune`](https://huggingface.co/patrickramos/bert-base-japanese-v2-wrime-fine-tune)  
を利用し、テキストおよび絵文字の情報から総合的な感情スコア（`combined_score`）を算出します。

テキストの感情強度（`text_sent_score`）は WRIME モデルによりロジット値から推定され、  
絵文字のスコア（`emoji_score`）と重み付きで統合されます。

---

## ⚙️ 使用モデル
- **モデル名**: `patrickramos/bert-base-japanese-v2-wrime-fine-tune`  
- **入力**: 日本語テキスト  
- **出力**: 8つの感情スコア  
  （joy, sadness, anticipation, surprise, anger, fear, disgust, trust）

---

## 🔧 ハイパーパラメータ

| パラメータ | 意味 | 既定値 |
|-------------|------|--------|
| `ALPHA` | 信頼度に基づく重み計算係数 | 0.5 |
| `BETA` | 重みのベース値 | 0.2 |
| `EPS` | 数値安定化用の微小値 | 1e-6 |
| `EMOJI_SCORE` | 絵文字ごとの感情スコア辞書 | ❤️:+0.9, 🙂:+0.6, 😢:-0.6, 😱:-0.9, 😖:-0.5, 🔥:+0.5, 😡:-0.8, 👍:+0.6 |

---

## 🧩 処理の流れ

### ① テキスト感情スコア（`text_sent_score`）
WRIME モデルから得られた各感情ロジット値をもとに、  
**log-sum-exp（LSE）ベース**でポジティブ・ネガティブの代表値を計算します。

```math
L_{pos} &= \log \sum_{i \in \{joy, trust, anticipation\}} e^{z_i} \\
L_{neg} &= \log \sum_{j \in \{sadness, anger, fear, disgust\}} e^{z_j} \\
p_{pos} &= \frac{1}{1 + e^{-(L_{pos} - L_{neg})}} \\
s_{text} &= 2p_{pos} - 1 \quad (-1 \leq s_{text} \leq 1)
```


また、モデル出力の信頼度 `c` は2クラス分布のエントロピーから算出します。

```math
c = 1 - \frac{H}{\log 2}, \quad H = -[p_{pos}\log p_{pos} + p_{neg}\log p_{neg}]
```

---

### ② 重み計算
テキストの信頼度 `c` に基づき、絵文字とテキストの比重を決定します。

```math
w_2 = clip(\alpha \cdot c + \beta, 0, 1)
\quad , \quad
w_1 = 1 - w_2
```

---

### ③ 総合感情スコア（`combined_score`）
絵文字スコアとテキストスコアを加重平均して総合スコアを得ます。

```math
s_{combined} = w_1 \cdot s_{emoji} + w_2 \cdot s_{text}
```

最終的に `[-1, 1]` の範囲を `0~100` に再スケーリングして出力します。

```math
s_{combined\_100} = \frac{(s_{combined} + 1)}{2} \times 100
```

---

## 🧮 出力例

```
--- 感情スコア結果 ---
text: 今日は友達と会えてとても嬉しい！
emoji: 🙂 
emoji_score: 0.6
text_sent_score: 1.0
confidence(c): 0.723
w1: 0.439
w2: 0.561
combined_score: 0.956
combined_score_100: 97.8
emotion_values: {'joy': 2.989, 'sadness': -0.044, 'anticipation': 0.231, 'surprise': 0.256, 'anger': -0.071, 'fear': 0.018, 'disgust': -0.035, 'trust': 0.172}
```

---

## 🧠 主な関数

| 関数名 | 説明 |
|---------|------|
| `get_text_sent_score(text)` | WRIMEモデルを用いてテキストの感情スコアと信頼度を算出 |
| `get_combined_score(text, emoji)` | テキストと絵文字を統合し、最終スコアを算出 |
| `emotion_values` | 各感情（joy/sadness等）の生ロジット値を格納 |

---

## 💡 応用例
- 日記・SNS投稿の感情可視化  
- 「感情×絵文字」相関データ分析  
- モバイルアプリでの感情トラッキング（例：HibiLog）
