import numpy as np

# Function to simulate the coin flips and check if B flips more heads than A
def simulate_flips(n, num_simulations=10000):
    b_more_heads = 0
    for _ in range(num_simulations):
        # Simulate A's n coin flips
        a_flips = np.random.binomial(n, 0.5)
        
        # Simulate B's n+2 coin flips
        b_flips = np.random.binomial(n+1, 0.5)
        
        # Check if B has more heads than A
        if b_flips > a_flips:
            b_more_heads += 1
    
    # Return the probability that B flips more heads than A
    return b_more_heads / num_simulations

# Run the simulation for n = 10
n = 100
prob_simulation = simulate_flips(n)
print(prob_simulation)
