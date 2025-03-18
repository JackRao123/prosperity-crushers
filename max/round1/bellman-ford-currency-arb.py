from math import log
import pandas as pd

def currency_to_index(currency):
    if currency == 'snowballs':
        return 0
    if currency == 'pizzas':
        return 1
    if currency == 'nuggets':
        return 2
    if currency == 'seashells':
        return 3
    
def index_to_currency(index):
    mapping = {0: 'snowballs', 1: 'pizzas', 2: 'nuggets', 3: 'seashells'}
    return mapping.get(index, "Unknown")

def bellman_ford(edges, n=4):
    dist = [float('inf')] * n
    parent = [-1] * n

    dist[0] = 0

    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u

    x = None 
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            parent[v] = u
            x = v
            break

    if x is None:
        print("No arbitrage opportunity detected.")
    else:
        for _ in range(n):
            x = parent[x]

        cycle_start = x
        cycle = [cycle_start]
        cur = parent[cycle_start]
        while cur != cycle_start:
            cycle.append(cur)
            cur = parent[cur]

        cycle.append(cycle_start)
        cycle.reverse()


        cycle_names = [index_to_currency(i) for i in cycle]
        print("Arbitrage cycle detected:")
        print(" -> ".join(cycle_names))

def main():
    currencies = ['snowballs', 'pizzas', 'nuggets', 'seashells']    # map to 0, 1, 2, 3
    exchange_matrix = [
        [1.0, 1.45, 0.52, 0.72],
        [0.7, 1.0, 0.31, 0.48],
        [1.95, 3.1, 1.0, 1.49],
        [1.34, 1.98, 0.64, 1.0]
    ]

    edges = []
    for i, row in enumerate(exchange_matrix):
        for j, rate in enumerate(row):
            weight = -log(rate)
            edges.append((i, j, weight))

    bellman_ford(edges)

    return 0

if __name__ == "__main__":
    main()