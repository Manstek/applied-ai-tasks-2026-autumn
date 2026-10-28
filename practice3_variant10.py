# -*- coding: utf-8 -*-
"""
Практическая работа 3. ВАРИАНТ 10
Субтокены и BIO, Token/Entity-level метрики, Weighted Cross-Entropy.

Дано:
  Часть 1: «неудовлетворительно» -> ['не', '##удовлет', '##ворительно'],
  сущность B-ADJ
  Часть 2: 5 истинных сущностей по 4 токена; предсказано 5;
           4 идеально, 1 с ошибкой границы, 0 пропущено;
           истинных токенов 20, угадано 19, лишних 1
  Часть 3: w1 = 100, y = 1, p = 0.1

Без внешних библиотек — только math.
"""

import math


def line(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


# ============================================================
# ЧАСТЬ 1. Субтокены и BIO
# ============================================================
subtokens = ["не", "##удовлет", "##ворительно"]
ENTITY = "ADJ"
IGNORE = -100   # метка «не учитывать в функции потерь»


# Нумерация стратегий — как в лекции 7 (слайды 4 и 22)

def strategy_1(subs, ent):
    """Стратегия 1: все субтокены получают метки — B на первом, I на остальных.
    Простая, но штрафует модель за продолжения."""
    return ["B-" + ent] + ["I-" + ent] * (len(subs) - 1)


def strategy_2(subs, ent):
    """Стратегия 2: в функции потерь учитывается только первый субтокен,
    остальные помечаются ignore_index = -100.
    На практике даёт лучшее качество."""
    return ["B-" + ent] + [IGNORE] * (len(subs) - 1)


def strategy_3(subs, ent):
    """Стратегия 3:
    все субтокены получают одинаковую метку B — нарушает BIO."""
    return ["B-" + ent] * len(subs)


def assemble_entities(subs, labels):
    """
    Сборка сущностей из BIO-последовательности.
    B открывает новую сущность, I продолжает текущую, O/-100 закрывает.
    """
    entities = []
    current = None
    for tok, lab in zip(subs, labels):
        if isinstance(lab, str) and lab.startswith("B-"):
            if current:
                entities.append(current)
            current = [tok]
        elif isinstance(lab, str) and lab.startswith("I-") and current:
            current.append(tok)
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)
    return entities


line("ЧАСТЬ 1. Субтокены и BIO")
print("Слово:", "«неудовлетворительно»")
print("Субтокены:", subtokens)
print("Истинная сущность: B-" + ENTITY)

print("\nВопрос 1. Стратегия 1 (B на первом субтокене, I на остальных):")
s1 = strategy_1(subtokens, ENTITY)
for t, l in zip(subtokens, s1):
    print(f"   {t:<16} -> {l}")
ents1 = assemble_entities(subtokens, s1)
print(f"   Сборка: {len(ents1)} сущность: {ents1}  — корректно")

print(
    "\nДля сравнения, Стратегия 2 (учитывается только первый,остальные -100):")
s2 = strategy_2(subtokens, ENTITY)
for t, l in zip(subtokens, s2):
    print(f"   {t:<16} -> {l}")
print("   Метка -100 = ignore_index: субтокен не участвует в функции потерь.")

print("\nВопрос 2. Стратегия 3 (всем B-" + ENTITY + "):")
s3 = strategy_3(subtokens, ENTITY)
for t, l in zip(subtokens, s3):
    print(f"   {t:<16} -> {l}")
ents3 = assemble_entities(subtokens, s3)
print(f"   Сборка: {len(ents3)} сущность(ей): {ents3}")
print("   Истинных сущностей: 1")
print(f"   ЛОЖНЫХ сущностей: {len(ents3) - 1}")


# ============================================================
# ЧАСТЬ 2. Метрики
# ============================================================
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


# --- данные варианта ---
true_entities = 5
tokens_per_entity = 4
pred_entities = 5
perfect = 4
boundary_errors = 1
missed = 0
true_tokens = 20
guessed_tokens = 19
extra_tokens = 1

line("ЧАСТЬ 2. Метрики")

# Вопрос 3 — Token-level
tp_t = guessed_tokens
fp_t = extra_tokens
fn_t = true_tokens - guessed_tokens
P_t, R_t, F_t = prf(tp_t, fp_t, fn_t)

print("Вопрос 3. Token-level")
print(f"   TP = {tp_t} (угадано)")
print(f"   FP = {fp_t} (лишний)")
print(f"   FN = {true_tokens} - {guessed_tokens} = {fn_t} (пропущен)")
print(f"   Precision = {tp_t} / ({tp_t} + {fp_t}) = {P_t:.4f}")
print(f"   Recall    = {tp_t} / ({tp_t} + {fn_t}) = {R_t:.4f}")
print(f"   F1        = 2*P*R / (P+R)        = {F_t:.4f}")

# Вопрос 4 — Entity-level
# сущность с ошибкой границы = одновременно FP (неверное предсказание)
#                             и FN (истинная не найдена)
tp_e = perfect
fp_e = pred_entities - perfect
fn_e = boundary_errors + missed
P_e, R_e, F_e = prf(tp_e, fp_e, fn_e)

print("\nВопрос 4. Entity-level (совпадение типа и обеих границ)")
print(f"   TP = {tp_e} (идеальные)")
print(f"   FP = {pred_entities} - {perfect} = {fp_e} (ошибка границы)")
print(f"   FN = {boundary_errors} (та же ошибка границы) + {missed} (пропущено) = {fn_e}")
print(f"   Precision = {tp_e} / ({tp_e} + {fp_e}) = {P_e:.4f}")
print(f"   Recall    = {tp_e} / ({tp_e} + {fn_e}) = {R_e:.4f}")
print(f"   F1        = 2*P*R / (P+R)      = {F_e:.4f}")

print(f"\n   Вывод: одна ошибка границы стоит {1 - F_t:.0%} F1 на уровне токенов")
print(f"          и {1 - F_e:.0%} F1 на уровне сущностей.")


# ============================================================
# ЧАСТЬ 3. Weighted Cross-Entropy
# ============================================================
w1 = 100
y = 1
p = 0.1

line("ЧАСТЬ 3. Weighted Cross-Entropy")

ln_p = math.log(p)
ce = -y * ln_p
wce = -w1 * y * ln_p

print("Вопрос 5.")
print("   Формула: L = -w * y * ln(p)")
print(f"   ln({p}) = {ln_p:.4f}")
print(f"   Weighted CE = -{w1} * {y} * ({ln_p:.4f}) = {wce:.2f}")
print(f"   Обычная CE  = -{y} * ({ln_p:.4f})       = {ce:.4f}")
print(f"   Штраф вырос в {wce / ce:.0f} раз")


# ============================================================
# ТАБЛИЦА ОТВЕТОВ
# ============================================================
line("ТАБЛИЦА ОТВЕТОВ")
print("1. Стратегия 1  :", ", ".join(f"{t}->{l}" for t, l in zip(
    subtokens, s1)))
print(f"2. Стратегия 3  : {len(
    ents3)} сущности вместо 1 -> {len(ents3) - 1} ложные")
print(f"3. Token-level  : P = {P_t:.2f}, R = {R_t:.2f}, F1 = {F_t:.2f}")
print(f"4. Entity-level : P = {P_e:.2f}, R = {R_e:.2f}, F1 = {F_e:.2f}")
print(f"5. Weighted CE  : {
    wce:.2f}  (без веса {ce:.2f}, штраф x{wce / ce:.0f})")
