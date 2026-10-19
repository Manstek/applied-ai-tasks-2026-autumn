# -*- coding: utf-8 -*-
"""
ВАРИАНТ 10 (Loss для всех позиций)
src: "We eat pizza"  ->  tgt: "Мы едим пиццу"
d_model = 4, h = 2, d_k = 2

Архитектура — ровно как в методичке:
  Энкодер:  X_attn   = X_src   + Z          (Z = Concat · W_O)
            X_out_enc = X_attn + FFN(X_attn)
  Декодер:  X_masked  = X_tgt   + Y_masked
            X_cross   = X_masked + Y_cross
            X_out_dec = X_cross  + Y_ffn
LayerNorm в этой схеме НЕ применяется.
"""

import math

d_model, h, d_k = 4, 2, 2


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]


def transpose(A):
    return [list(c) for c in zip(*A)]


def add(A, B):
    return [[a + b for a, b in zip(x, y)] for x, y in zip(A, B)]


def hstack(ms):
    return [sum((m[i] for m in ms), []) for i in range(len(ms[0]))]


def show(name, A):
    print(f"{name} =")
    for row in A:
        print("   [" + "  ".join(f"{v:8.4f}" for v in row) + "]")


def softmax(v):
    mx = max(v); e = [math.exp(x - mx) for x in v]; s = sum(e)
    return [x / s for x in e]


# ============ ДАНО ============
src_tokens = ["We", "eat", "pizza", "<EOS>"]
E_src = [[0.5,  0.8, -0.3, 0.1],    # We
         [1.2,  0.5,  0.6, 0.9],    # eat
         [0.9, -0.1, -1.1, 0.5],    # pizza
         [0.5,  0.5,  0.5, 0.5]]    # <EOS>

tgt_vocab = ["<SOS>", "Мы", "едим", "пиццу", "<EOS>"]
E_tgt = [[0.0,  0.0,  0.0, 0.0],    # <SOS>
         [0.4,  0.7, -0.2, 0.1],    # Мы
         [1.3,  0.5,  0.7, 1.0],    # едим
         [0.8, -0.1, -1.0, 0.6],    # пиццу
         [1.0,  1.0,  1.0, 1.0]]    # <EOS>

# PE взято из задания (вариант 1), в том же округлении
PE = [[0.00,  1.00, 0.00, 1.0],     # поз. 0
      [0.84,  0.54, 0.01, 1.0],     # поз. 1
      [0.91, -0.42, 0.02, 1.0],     # поз. 2
      [0.14, -0.99, 0.03, 1.0]]     # поз. 3

dec_in_ids = [0, 1, 2, 3]           # <SOS> Мы едим пиццу
target_ids = [1, 2, 3, 4]           # Мы едим пиццу <EOS>

# ---- веса внимания (из задания) ----
W_Q1 = [[0.2, 0.1],
        [0.1, 0.3],
        [0.4, -0.1],
        [0.0,  0.2]]
W_K1 = [[0.1, 0.2],
        [0.3, 0.0],
        [0.1,  0.4],
        [0.2, -0.1]]
W_V1 = [[0.0, 0.2],
        [0.3, 0.1],
        [0.2,  0.0],
        [0.1,  0.3]]

W_Q2 = [[0.1, 0.0], [0.2, 0.2], [0.0,  0.3], [0.3,  0.1]]
W_K2 = [[0.2, 0.1], [0.0, 0.3], [0.3, -0.1], [0.1,  0.2]]
W_V2 = [[0.1, 0.1], [0.2, 0.3], [0.0,  0.2], [0.3,  0.0]]

HEADS = [(W_Q1, W_K1, W_V1), (W_Q2, W_K2, W_V2)]

W_O = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]   # I4

# ---- FFN: в варианте 10 не задан, выбран самостоятельно ----
W1 = [[0.1, -0.2,  0.3,  0.0,  0.2,  0.1, -0.1,  0.2],
      [0.2,  0.3, -0.1,  0.1,  0.0,  0.2,  0.3, -0.2],
      [-0.1,  0.1,  0.2,  0.3, -0.2,  0.0,  0.1,  0.1],
      [0.3,  0.0,  0.1, -0.2,  0.1,  0.3,  0.0,  0.2]]

W2 = [[0.2, -0.1,  0.3,  0.0],
      [0.1,  0.2, -0.2,  0.1],
      [0.0,  0.3,  0.1, -0.1],
      [0.2,  0.0, -0.1,  0.3],
      [-0.1,  0.2,  0.2,  0.0],
      [0.3, -0.2,  0.0,  0.1],
      [0.1,  0.1,  0.2, -0.2],
      [0.0,  0.2, -0.1,  0.3]]

# ---- выходной слой (строка = слово русского словаря) ----
W_head = [[0.1,  0.6, -0.2,  0.0],   # <SOS>
          [0.8, -0.1,  0.5,  0.9],   # Мы
          [1.1,  0.4,  0.7, -0.2],   # едим
          [0.2,  1.0, -0.4,  0.5],   # пиццу
          [-0.3,  0.0,  0.5,  1.1]]   # <EOS>


# ============ блоки ============
def attention(Q, K, V, masked=False, label=""):
    scores = [[s / math.sqrt(d_k) for s in r] for r in matmul(Q, transpose(K))]
    if label:
        show(f"   {label}: Scores", scores)
    if masked:
        for i in range(len(scores)):
            for j in range(len(scores[0])):
                if j > i:
                    scores[i][j] = -1e9
    A = [softmax(r) for r in scores]
    if label:
        show(f"   {label}: Softmax", A)
    return matmul(A, V)


def multi_head(X_q, X_kv, masked=False, name="", verbose=False):
    heads = []
    for idx, (Wq, Wk, Wv) in enumerate(HEADS, 1):
        Q, K, V = matmul(X_q, Wq), matmul(X_kv, Wk), matmul(X_kv, Wv)
        if verbose:
            show(f"   {name} | голова {idx}: Q", Q)
            show(f"   {name} | голова {idx}: K", K)
            show(f"   {name} | голова {idx}: V", V)
        heads.append(attention(Q, K, V, masked, f"{name} | голова {idx}"))
        show(f"   {name} | голова {idx}: Head{idx}", heads[-1])
    concat = hstack(heads)
    show(f"   {name}: Concat", concat)
    return matmul(concat, W_O)


def ffn(X, label=""):
    H = [[max(0.0, v) for v in row] for row in matmul(X, W1)]   # ReLU
    if label:
        show(f"   {label}: h = ReLU(X·W1)", H)
    return matmul(H, W2)


# ================= ШАГ 1 =================
print("=" * 64)
print("ШАГ 1. X_src и X_tgt")
print("=" * 64)
X_src = add(E_src, PE)
X_tgt = add([E_tgt[i] for i in dec_in_ids], PE)
show("X_src (4x4)", X_src)
show("X_tgt (4x4)", X_tgt)

# ================= ШАГ 2 =================
print("\n" + "=" * 64)
print("ШАГ 2. Энкодер")
print("=" * 64)
Z = multi_head(X_src, X_src, name="Self-Attn энкодера", verbose=True)
show("Z = Concat · W_O", Z)
X_attn = add(X_src, Z)
show("X_attn = X_src + Z", X_attn)
FFN_out = ffn(X_attn, label="FFN энкодера")
show("   FFN энкодера: FFN_out = h · W2", FFN_out)
X_out_enc = add(X_attn, FFN_out)
show("X_out_enc = X_attn + FFN_out", X_out_enc)

# ================= ШАГ 3 =================
print("\n" + "=" * 64)
print("ШАГ 3. Декодер (позиции 0, 1, 2, 3)")
print("=" * 64)
Y_masked = multi_head(X_tgt, X_tgt, masked=True, name="Masked Self-Attn")
X_masked = add(X_tgt, Y_masked)
show("X_masked = X_tgt + Y_masked", X_masked)

Y_cross = multi_head(X_masked, X_out_enc, name="Cross-Attn")
X_cross = add(X_masked, Y_cross)
show("X_cross = X_masked + Y_cross", X_cross)

Y_ffn = ffn(X_cross, label="FFN декодера")
show("   FFN декодера: FFN_out = h · W2", Y_ffn)
X_out_dec = add(X_cross, Y_ffn)
show("X_out_dec = X_cross + Y_ffn", X_out_dec)

# ================= ШАГ 4 =================
print("\n" + "=" * 64)
print("ШАГ 4. Logits, Softmax, Loss по позициям")
print("=" * 64)
logits = matmul(X_out_dec, transpose(W_head))
losses = []
for i, row in enumerate(logits):
    p = softmax(row)
    t = target_ids[i]
    loss_i = -math.log(p[t])
    losses.append(loss_i)
    print(f"\n--- Позиция {i}: вход '{
        tgt_vocab[dec_in_ids[i]]}' -> цель '{tgt_vocab[t]}' ---")
    print("  Logits :", "  ".join(f"{v:7.4f}" for v in row))
    print("  Softmax:")
    for w, pv in zip(tgt_vocab, p):
        print(f"      P({w:6s}) = {pv:.4f}")
    best = max(range(len(p)), key=lambda k: p[k])
    print(f"  Предсказано: '{tgt_vocab[best]}'")
    print(f"  Loss_{i} = -ln(P('{
        tgt_vocab[t]}')) = -ln({p[t]:.4f}) = {loss_i:.4f}")

# ================= ШАГ 5 =================
mean_loss = sum(losses) / len(losses)
print("\n" + "=" * 64)
print("ШАГ 5. Средний Loss")
print("=" * 64)
print("Средний Loss = (" + " + ".join(f"{l:.4f}" for l in losses) + ") / 4")
print(f"Средний Loss = {mean_loss:.4f}")
print(f"Perplexity   = {math.exp(mean_loss):.4f}")
