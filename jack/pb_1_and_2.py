import matplotlib.pyplot as plt
import numpy as np
import heapq
from collections import defaultdict
from tqdm import tqdm

# 1) Prepare data copies
md3 = market_data_round_3_all3days.copy()
th3 = trades_round_3_all3days.copy()

# 2) Filter by instrument
croiss_df = md3[md3["product"] == "CROISSANTS"].reset_index(drop=True)
jam_df     = md3[md3["product"] == "JAMS"].reset_index(drop=True)
djem_df    = md3[md3["product"] == "DJEMBES"].reset_index(drop=True)
b1_df      = md3[md3["product"] == "PICNIC_BASKET1"].reset_index(drop=True)
b2_df      = md3[md3["product"] == "PICNIC_BASKET2"].reset_index(drop=True)

# 3) Build synthetic basket values
syn1 = 6 * croiss_df["mid_price"] + 3 * jam_df["mid_price"] + djem_df["mid_price"]
syn2 = 4 * croiss_df["mid_price"] + 2 * jam_df["mid_price"]

# 4) Construct augmented & artificial valuations
aug1 = b2_df["mid_price"] + 2 * croiss_df["mid_price"] + jam_df["mid_price"] + djem_df["mid_price"]
art1 = 1.5 * b2_df["mid_price"] + djem_df["mid_price"]

# 5) Compute spreads
sp1    = b1_df["mid_price"] - syn1
sp2    = b2_df["mid_price"] - syn2
sp1aug = b1_df["mid_price"] - aug1
sp1art = b1_df["mid_price"] - art1

# 6) Plot all spreads
plt.figure(figsize=(12, 6))
plt.plot(b1_df["timestamp"], sp1,    label="B1 − (6C+3J+D)")
plt.plot(b1_df["timestamp"], sp2,    label="B2 − (4C+2J)")
plt.plot(b1_df["timestamp"], sp1aug, label="B1 − augmented")
plt.plot(b1_df["timestamp"], sp1art, label="B1 − artificial")
plt.title("Spreads Between Basket 1 and Synthetic Constructs")
plt.xlabel("Timestamp")
plt.ylabel("Spread")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7) Search for optimal hedges via best-first search
#    Node format: (c, d, j, b1, b2, C, D, J, B1, B2)
LIMITS = (-250, -60, -350, -60, -100, 250, 60, 350, 60, 100)

# Moves: delta positions for each trade action
trade_moves = {
    "long_b1_short_nav":                      (-6, -1, -3,  0,  0,  0,  0,  0, 1, 0),
    "short_b1_long_nav":                      ( 0,  0,  0, -1,  0,  6,  1,  3, 0, 0),
    "long_b2_short_nav":                      (-4,  0, -2,  0,  0,  0,  0,  0, 0, 1),
    "short_b2_long_nav":                      ( 0,  0,  0,  0, -1,  4,  0,  2, 0, 0),
    "long_b1_short_b2nav_plus":               (-2, -1, -1, 0, -1,  0,  0,  0, 1, 0),
    "short_b1_long_b2nav_plus":               ( 0,  0,  0, -1,  0,  2,  1,  1, 0, 1),
    "long_b1_short_b2nav_mul":                ( 0, -1,  0,  0, -1.5,0,  0,  0, 1, 0),
    "short_b1_long_b2nav_mul":                ( 0,  0,  0, -1,  0,  0,  1,  0, 0,1.5),
    "long_croissant":                         ( 0,  0,  0,  0,  0,  1,  0,  0, 0, 0),
    "short_croissant":                        ( 0,  0,  0,  0,  0, -1,  0,  0, 0, 0),
    "long_djembe":                            ( 0,  0,  0,  0,  0,  0,  1,  0, 0, 0),
    "short_djembe":                           ( 0,  0,  0,  0,  0,  0, -1,  0, 0, 0),
    "long_jam":                               ( 0,  0,  0,  0,  0,  0,  0,  1, 0, 0),
    "short_jam":                              ( 0,  0,  0,  0,  0,  0,  0, -1, 0, 0),
}

# Reverse lookup for reconstruction
move_names = {delta: name for name, delta in trade_moves.items()}

# Imbalance deltas (C, D, J)
imbalance_deltas = {
    "long_b1_short_nav":                      (0, 0, 0),
    "short_b1_long_nav":                      (0, 0, 0),
    "long_b2_short_nav":                      (0, 0, 0),
    "short_b2_long_nav":                      (0, 0, 0),
    "long_b1_short_b2nav_plus":               (0, 0, 0),
    "short_b1_long_b2nav_plus":               (0, 0, 0),
    "long_b1_short_b2nav_mul":                (0, 0, 0),
    "short_b1_long_b2nav_mul":                (0, 0, 0),
    "long_croissant":                         (1, 0, 0),
    "short_croissant":                        (-1,0, 0),
    "long_djembe":                            (0, 1, 0),
    "short_djembe":                           (0,-1, 0),
    "long_jam":                               (0, 0, 1),
    "short_jam":                              (0, 0,-1),
}

def in_bounds(state):
    """Ensure each element respects its min/max limits."""
    return all(l <= state[i] <= LIMITS[i+5] for i, l in enumerate(LIMITS[:5]))

def vec_add(a, b):
    return tuple(a[i] + b[i] for i in range(len(a)))

def vec_sub(a, b):
    return tuple(a[i] - b[i] for i in range(len(a)))

def imb_penalty(imb):
    return sum(abs(x) for x in imb)

def find_strategy(t_idx, start=None, max_nodes=20000, hedge_pen=1):
    """Best-first search over trade_moves to maximize profit minus hedge penalty."""
    if start is None:
        start = {}
    origin = (
        start.get("CROISSANTS",0), start.get("DJEMBES",0), start.get("JAMS",0),
        start.get("PICNIC_BASKET1",0), start.get("PICNIC_BASKET2",0),
        start.get("CROISSANTS",0), start.get("DJEMBES",0), start.get("JAMS",0),
        start.get("PICNIC_BASKET1",0), start.get("PICNIC_BASKET2",0),
    )
    origin_imb = (
        start.get("CROISSANTS",0) + 6*start.get("PICNIC_BASKET1",0) + 4*start.get("PICNIC_BASKET2",0),
        start.get("DJEMBES",0)   +    start.get("PICNIC_BASKET1",0),
        start.get("JAMS",0)      + 3*start.get("PICNIC_BASKET1",0) + 2*start.get("PICNIC_BASKET2",0),
    )

    scores = {origin: -hedge_pen * imb_penalty(origin_imb)}
    imbalances = {origin: origin_imb}
    parents = {}
    seen = {origin}
    heap = [(-scores[origin], origin)]

    while heap and len(seen) < max_nodes:
        val, node = heapq.heappop(heap)
        for mv, delta in trade_moves.items():
            nxt = vec_add(node, delta)
            if not in_bounds(nxt):
                continue
            imb_u = imbalances[node]
            imb_v = vec_add(imb_u, imbalance_deltas[mv])
            new_score = scores[node] + profitability[mv][t_idx] - hedge_pen * (imb_penalty(imb_v) - imb_penalty(imb_u))
            if new_score > scores.get(nxt, float("-inf")):
                scores[nxt] = new_score
                imbalances[nxt] = imb_v
                parents[nxt] = delta
                seen.add(nxt)
                heapq.heappush(heap, (-new_score, nxt))

    best_node, best_val = max(scores.items(), key=lambda x: x[1])
    # Reconstruct move counts
    counts = defaultdict(int)
    cur = best_node
    while cur != origin:
        d = parents[cur]
        counts[move_names[d]] += 1
        cur = vec_sub(cur, d)
    return counts, best_val

# 8) Run over sample timestamps
for t in range(10000):
    strat, prof = find_strategy(t, start={"CROISSANTS":-118, "DJEMBES":48, "JAMS":-136}, max_nodes=20000, hedge_pen=5)
    print(strat, prof)

# 9) Test impact of search depth
depth_results = defaultdict(int)
for max_n in tqdm(np.linspace(100,1400,20,dtype=int)):
    for t in np.linspace(0,9999,400,dtype=int):
        _, val = find_strategy(t, start={"CROISSANTS":200}, max_nodes=max_n, hedge_pen=200)
        depth_results[max_n] += val
depth_results
