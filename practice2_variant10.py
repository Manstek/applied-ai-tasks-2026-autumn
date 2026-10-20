# -*- coding: utf-8 -*-
"""
Практическая работа 2. ВАРИАНТ 10
Декодирование LLM: температура, top-p, штраф за повторы, метрики.

Дано:
  z1 = P5 = [2.5, 2.4, 0.5, 0.4, 0.3],  T = 1.0
  фильтр: top-p 0.9,  выбор: жадный
  z2 = [1.4, 2.6, 0.8, 1.9, 1.1],  ро = 1.5
  TTFT = 0.50 c, TPOT = 0.040 c, N = 128
  KV-кэш: L = 32, H_kv = 8, D_head = 128, S = 2048, B = 1, 2 байта

"""

import math

# ============================================================
# ИСХОДНЫЕ ДАННЫЕ ВАРИАНТА 10
# ============================================================
tokens = ["A", "B", "C", "D", "E"]

z1 = [2.5, 2.4, 0.5, 0.4, 0.3]      # логиты первого шага (набор P5)
T = 1.0                              # температура
TOP_P = 0.9                          # порог ядра
GREEDY = True                        # выбор жадный (u не нужен)

z2 = [1.4, 2.6, 0.8, 1.9, 1.1]      # логиты второго шага
RHO = 1.5                            # штраф за повторы

TTFT = 0.50                          # время до первого токена, с
TPOT = 0.040                         # время на токен, с
N = 128                              # число токенов

L = 32                               # слоёв
H_KV = 8                             # kv-голов
D_HEAD = 128                         # размер головы
S = 2048                             # длина контекста
B = 1                                # размер пакета
BYTES = 2                            # байт на число


def line(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# ШАГ 1. Softmax с температурой (трюк вычитания максимума)
# ============================================================
line("ШАГ 1. Вероятности при температуре T = %.1f" % T)

scaled = [z / T for z in z1]
print("z1      :", "  ".join(f"{v:6.3f}" for v in z1))
print("z1 / T  :", "  ".join(f"{v:6.3f}" for v in scaled))

mx = max(scaled)
shifted = [v - mx for v in scaled]
print("минус max (%.3f):" % mx, "  ".join(f"{v:6.3f}" for v in shifted))

exps = [math.exp(v) for v in shifted]
total = sum(exps)
probs = [e / total for e in exps]

print(f"\n{'Токен':<8}{'z/T - max':>12}{'Экспонента':>14}{'Вероятность':>14}")
print("-" * 48)
for t, s, e, p in zip(tokens, shifted, exps, probs):
    print(f"{t:<8}{s:>12.4f}{e:>14.4f}{p:>14.4f}")
print("-" * 48)
print(f"{'сумма':<8}{'':>12}{total:>14.4f}{sum(probs):>14.4f}")
print("\nКонтроль: сумма вероятностей = %.4f" % sum(probs))


# ============================================================
# ШАГ 2. Фильтр top-p и перенормировка
# ============================================================
line("ШАГ 2. Фильтр top-p %.2f" % TOP_P)

# сортируем по убыванию вероятности, запоминая исходные индексы
order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

nucleus = []
cum = 0.0
print("Накопленная сумма:")
for i in order:
    cum += probs[i]
    nucleus.append(i)
    mark = "ядро закрывается" if cum >= TOP_P else "продолжаем"
    print(f"  + {tokens[i]}  ->  {cum:.4f}   ({mark})")
    if cum >= TOP_P:
        break

nuc_sum = sum(probs[i] for i in nucleus)
print(f"\nЯдро: {{{', '.join(tokens[i] for i in nucleus)}}}, сумма = {
    nuc_sum:.4f}")

renorm = {i: probs[i] / nuc_sum for i in nucleus}

print(f"\n{'Токен':<8}{'было':>12}{'стало':>12}")
print("-" * 32)
for i in nucleus:
    print(f"{tokens[i]:<8}{probs[i]:>12.4f}{renorm[i]:>12.4f}")
print("-" * 32)
print(f"{'сумма':<8}{nuc_sum:>12.4f}{sum(renorm.values()):>12.4f}")

dropped = [tokens[i] for i in range(len(tokens)) if i not in nucleus]
if dropped:
    print("\nОтброшены:", ", ".join(dropped))


# ============================================================
# ШАГ 3. Выбор токена
# ============================================================
line("ШАГ 3. Выбор токена (жадно)" if GREEDY else "ШАГ 3. Выбор токена")

chosen = max(nucleus, key=lambda i: renorm[i])
print("Жадный выбор = максимум вероятности в ядре.")
print(f"Выбран токен '{tokens[chosen]}', индекс {chosen}")
print(f"Вероятность в ядре: {renorm[chosen]:.4f}")
print(f"Вероятность до перенормировки: {probs[chosen]:.4f}")


# ============================================================
# ШАГ 4. Второй шаг со штрафом за повторы
# ============================================================
line("ШАГ 4. Штраф за повторы (ро = %.1f)" % RHO)

print("z2          :", "  ".join(f"{v:6.3f}" for v in z2))
print(f"Штраф применяется к индексу {chosen} (токен '{tokens[chosen]}') — "
      f"выбранному на шаге 3.")

old = z2[chosen]
if old > 0:
    new = old / RHO
    rule = f"логит положительный ({old}) -> делим: {old} / {RHO} = {new:.4f}"
else:
    new = old * RHO
    rule = f"логит неположительный ({old}) -> умножаем: {old} * {
        RHO} = {new:.4f}"
print("Правило:", rule)

z2_mod = list(z2)
z2_mod[chosen] = new
print("z2 со штрафом:", "  ".join(f"{v:6.3f}" for v in z2_mod))

best_with = max(range(len(z2_mod)), key=lambda i: z2_mod[i])
best_without = max(range(len(z2)), key=lambda i: z2[i])

print(f"\nЖадный выбор СО штрафом : '{tokens[best_with]}'    (логит {z2_mod[
    best_with]:.4f})")
print(f"Жадный выбор БЕЗ штрафа : '{tokens[best_without]}'    (логит {z2[
    best_without]:.4f})")

if best_with != best_without:
    print("\nВЫВОД: штраф развернул решение — повтор предотвращён.")
else:
    print("\nВЫВОД: разворота решения НЕ произошло.")
    print(f"Оштрафованный токен '{tokens[
        chosen]}' и без штрафа не был лидером")
    print(f"(его логит {z2[chosen]} против {z2[best_without]} у '{tokens[
        best_without]}'),")
    print("поэтому штраф на исход не повлиял.")


# ============================================================
# ШАГ 5. Метрики и память
# ============================================================
line("ШАГ 5. Метрики и память KV-кэша")

latency = TTFT + TPOT * (N - 1)
throughput = N / latency

print("Latency = TTFT + TPOT * (N - 1)")
print(f"        = {TTFT} + {TPOT} * {N - 1}")
print(f"        = {TTFT} + {TPOT * (N - 1):.2f}")
print(f"        = {latency:.2f} с")

print(f"\nThroughput = N / Latency = {N} / {latency:.2f} = "
      f"{throughput:.1f} ток/с")

mem = 2 * L * H_KV * D_HEAD * S * B * BYTES
print("\nM = 2 * L * H_kv * D_head * S * B * байт")
print(f"  = 2 * {L} * {H_KV} * {D_HEAD} * {S} * {B} * {BYTES}")
print(f"  = {mem:,} байт".replace(",", " "))
print(f"  = {mem / 1024:,.0f} КиБ".replace(",", " "))
print(f"  = {mem / 1024 / 1024:.0f} МиБ")


# ============================================================
# ИТОГОВАЯ ТАБЛИЦА ОТВЕТОВ
# ============================================================
line("ТАБЛИЦА ОТВЕТОВ")

print("1. Вероятности        :", "[" + ", ".join
      (f"{p:.4f}" for p in probs) + "]")
print(f"                        сумма {sum(probs):.4f}")
print(f"2. Ядро и веса        : {{{', '.join(tokens[i] for i in nucleus)}}}, "
      f"сумма {nuc_sum:.4f}")
print("                        [" +
      ", ".join(f"{renorm[i]:.4f}" for i in nucleus) + "]")
print(f"3. Выбранный токен    : {tokens[chosen]} (индекс {chosen}), "
      f"p = {renorm[chosen]:.4f}")
print("4. Логиты со штрафом  : [" + ", ".join
      (f"{v:.4f}" for v in z2_mod) + "]")
print(f"                        со штрафом — {tokens[best_with]}, "
      f"без штрафа — {tokens[best_without]}")
print(f"                        разворот: {'да' if best_with != best_without else 'нет'}")
print(f"5. Latency/Throughput : {latency:.2f} с / {throughput:.1f} ток/с")
print(f"6. Память KV-кэша     : {mem / 1024 / 1024:.0f} МиБ")
