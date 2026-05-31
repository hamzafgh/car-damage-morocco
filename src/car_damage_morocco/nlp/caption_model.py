"""CNN + LSTM image-captioning model (no LLM, no RAG, no transformer).

Architecture
------------
  encoder : MobileNetV3-Small (ImageNet pretrained, frozen) -> 1024-d image features
  decoder : Embedding -> LSTM (256) -> Dense(vocab) over a word-level French vocab
  fusion  : image feature is concatenated with the start token's embedding
            (Show-and-Tell-style injection — simple, trains fast on a T4)

Inputs at training time:
  image   : (H, W, 3) uint8, resized to 224x224 in the preprocessor
  caption : sequence of token ids ending with <end>
Loss: sparse categorical cross-entropy with masking on <pad>.

Inference:
  greedy_decode(model, image, vocab, max_len=30) -> French string.

This module only defines the architecture and the encoder preprocessor.
Training is in train_caption_model.py.
"""
from __future__ import annotations
from typing import Sequence
import tensorflow as tf
from tensorflow.keras import layers, Model

IMG_SIZE = 224
FEAT_DIM = 1024            # MobileNetV3-Small global-pool output
EMBED_DIM = 256
LSTM_UNITS = 256


# ----------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------
def build_image_encoder(trainable: bool = False) -> Model:
    """Frozen MobileNetV3-Small -> 1024-d feature vector."""
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
        include_preprocessing=True,  # expects uint8 input in [0,255]
    )
    base.trainable = trainable
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype="uint8", name="image")
    x = tf.cast(inp, tf.float32)
    feat = base(x, training=False)
    feat = layers.Dense(EMBED_DIM, activation="relu", name="img_proj")(feat)
    return Model(inp, feat, name="image_encoder")


# ----------------------------------------------------------------------
# Decoder (training graph: teacher-forced)
# ----------------------------------------------------------------------
def build_captioner(vocab_size: int) -> Model:
    """Returns the full training model: (image, input_tokens) -> next_token logits."""
    image_in = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype="uint8", name="image")
    tokens_in = layers.Input(shape=(None,), dtype="int32", name="tokens_in")

    encoder = build_image_encoder(trainable=False)
    img_feat = encoder(image_in)                              # (B, EMBED_DIM)
    img_feat = layers.RepeatVector(1)(img_feat)               # (B, 1, EMBED_DIM)

    embed = layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True, name="embed")
    tok_emb = embed(tokens_in)                                # (B, T, EMBED_DIM)

    # Prepend the image-feature "token" to the sequence
    seq = layers.Concatenate(axis=1)([img_feat, tok_emb])     # (B, T+1, EMBED_DIM)
    lstm_out = layers.LSTM(LSTM_UNITS, return_sequences=True, name="lstm")(seq)
    # Drop the first step (it corresponded to the image-feature prefix)
    lstm_out = layers.Lambda(lambda t: t[:, 1:, :], name="strip_img_step")(lstm_out)
    logits = layers.Dense(vocab_size, name="logits")(lstm_out)

    model = Model([image_in, tokens_in], logits, name="cnn_lstm_captioner")
    return model


# ----------------------------------------------------------------------
# Loss with <pad> masking
# ----------------------------------------------------------------------
def masked_sparse_ce(pad_id: int = 0):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
        per_tok = scce(y_true, y_pred) * mask
        return tf.reduce_sum(per_tok) / tf.maximum(tf.reduce_sum(mask), 1.0)
    return loss


# ----------------------------------------------------------------------
# Greedy inference
# ----------------------------------------------------------------------
def greedy_decode(
    model: Model,
    image: tf.Tensor,            # uint8, (H, W, 3) or (1, H, W, 3)
    vocab: Sequence[str],
    start_id: int,
    end_id: int,
    max_len: int = 30,
) -> str:
    if image.shape.rank == 3:
        image = tf.expand_dims(image, 0)
    image = tf.image.resize(tf.cast(image, tf.float32), (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.uint8)

    tokens = tf.constant([[start_id]], dtype=tf.int32)        # (1, 1)
    out_ids: list[int] = []
    for _ in range(max_len):
        logits = model([image, tokens], training=False)        # (1, T, V)
        next_id = int(tf.argmax(logits[0, -1, :]).numpy())
        if next_id == end_id:
            break
        out_ids.append(next_id)
        tokens = tf.concat([tokens, [[next_id]]], axis=1)
    return " ".join(vocab[i] for i in out_ids)


if __name__ == "__main__":
    m = build_captioner(vocab_size=200)
    m.summary(line_length=110)
