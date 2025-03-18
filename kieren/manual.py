import numpy as np
import itertools
import math

conversion = np.array(
    [[1, 1.45, 0.52, 0.72],
    [0.7, 1, 0.31, 0.48],
    [1.95, 3.1, 1, 1.49],
    [1.34, 1.98, 0.64, 1]
])
products = {0:'Snowball', 1:'Pizza', 2:'Silicon Nugget', 3:'Seashell'}

def amount(seq):
    if not seq:
        return 1
    prod = conversion[3, seq[0]] * conversion[seq[-1], 3]
    L = len(seq)
    for i in range(L-1):
        prod *= conversion[seq[i], seq[i+1]]
    return prod

  
def maximise(L):
    # get all possible sequences
    seqs = itertools.product(*[range(0, 4) for _ in range(L)])
    max = 0
    for seq in seqs:
        p = amount(seq)
        if p > max:
            max = p
            res = seq
    return (res, max)

# # can only make 4 trades before having to come back to seashell
# r = maximise(4)
# trades = [products[3]] + [products[i] for i in r] + [products[3]]
# print(', '.join(trades))

results = []

for i in range(4):
    results.append(maximise(i))

print(results)
