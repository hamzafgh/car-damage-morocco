# CNN+LSTM captioner — `caption_model.py`

End-to-end image → French sentence neural network. Show-and-Tell style.

Module: `src/car_damage_morocco/nlp/caption_model.py`.

## Architecture at a glance

```
   image (224×224×3 uint8)
        │
        ▼
   ┌─────────────────────────────┐
   │  MobileNetV3-Small (frozen) │   ImageNet pretrained
   │  global avg pool → 1024-d   │
   └────────────────┬────────────┘
                    │
                    ▼
              Dense(256, relu)        ← img_proj
                    │
                    ▼ (B, 1, 256)     ← RepeatVector(1)
              ┌─────┴─────┐
              │  Concat   │  ◄── token_embeddings (B, T, 256)
              └─────┬─────┘            from Embedding(vocab, 256, mask_zero=True)
                    ▼ (B, T+1, 256)
              ┌───────────┐
              │  LSTM 256 │   return_sequences=True
              └─────┬─────┘
                    ▼ (B, T+1, 256)
              strip first step  (Lambda t: t[:, 1:, :])
                    │ (B, T, 256)
                    ▼
              Dense(vocab_size)        ← per-token logits
```

| Constant | Value |
|---|---|
| `IMG_SIZE` | 224 |
| `FEAT_DIM` | 1024 (MobileNetV3-Small GAP output) |
| `EMBED_DIM` | 256 |
| `LSTM_UNITS` | 256 |

## Why MobileNetV3-Small

- **Small** — ~2.5M backbone params, total model under 5M.
- **Built-in preprocessing** (`include_preprocessing=True`) — accepts `uint8 [0, 255]` directly.
- **T4-friendly** — trains end-to-end on free Kaggle GPU in well under an hour.
- **Deployable** — model could run on mobile (TFLite) if you wanted an offline mode for field inspectors.

## Why Show-and-Tell injection

The image feature is prepended as the very first "token" of the decoder sequence, then stripped from the output. This is the canonical injection scheme from *Vinyals et al. 2015* — simple, well-understood, and avoids the complexity of attention mechanisms.

You could swap for an attention-based decoder (Show-Attend-Tell, transformer) for a small accuracy bump, but at the cost of a heavier model that needs more data.

## Training via knowledge distillation

The clever bit: **there is no hand-labeled French caption dataset**.

The training corpus is generated **synthetically** from the [template engine](templates.md):

1. Run the **Stage 0/1/2 pipeline** on a batch of car images → get structured outputs (part, damage, area, position).
2. Feed those into `report()` from `describe_damage.py` → get a French sentence per image.
3. Train the CNN+LSTM to predict that sentence from the **image alone**.

This is **knowledge distillation**: the template engine is the *teacher*, the network is the *student*. The student learns to map image → text in one forward pass without needing the multi-stage pipeline at inference.

## Loss with `<pad>` masking

Sentences in a batch have different lengths → pad shorter ones with `<pad>` (id 0). Mask the pad positions so they don't contribute to the loss:

```python
def masked_sparse_ce(pad_id: int = 0):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
        per_tok = scce(y_true, y_pred) * mask
        return tf.reduce_sum(per_tok) / tf.maximum(tf.reduce_sum(mask), 1.0)
    return loss
```

Standard seq2seq trick. The `Embedding(... mask_zero=True)` already propagates the mask through the LSTM, but we re-apply it at the loss to be explicit.

## Greedy inference

```python
def greedy_decode(model, image, vocab, start_id, end_id, max_len=30):
    tokens = tf.constant([[start_id]], dtype=tf.int32)
    out_ids = []
    for _ in range(max_len):
        logits = model([image, tokens], training=False)
        next_id = int(tf.argmax(logits[0, -1, :]).numpy())
        if next_id == end_id:
            break
        out_ids.append(next_id)
        tokens = tf.concat([tokens, [[next_id]]], axis=1)
    return " ".join(vocab[i] for i in out_ids)
```

Per step: feed `(image, tokens_so_far)`, take argmax of the last-step logits, append. Stop on `<end>` or `max_len`.

You could swap greedy for **beam search** (top-k branches at each step) for a small quality bump. Greedy is sufficient for short, repetitive descriptions like these.

## Build + training entry point

`scripts/train_caption_model.py` ties together:

1. Loading the synthetic caption corpus
2. Building the word-level vocabulary
3. Tokenizing
4. `build_captioner(vocab_size)` → the Keras model
5. `model.compile(loss=masked_sparse_ce(), optimizer=Adam(1e-3))`
6. `model.fit(...)` with teacher forcing
7. Saving weights

## Defense angle

If a juror asks *"why have both a template engine and a captioner?"*:

> The template engine is what I'd ship in an insurance product — deterministic, grammatically correct, auditable. The captioner is what proves I understand representation learning end-to-end: image → French via a single forward pass, trained from data, no rules. Together they cover the engineering and the ML sides of the curriculum.
