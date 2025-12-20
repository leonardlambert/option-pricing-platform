import numpy as np

def calculate_strikes(spot, central_strike):
    rel_offsets = [0.85, 0.95, 1.05, 1.15]
    neighbors = [int(np.ceil((spot * r) / 5.0) * 5) for r in rel_offsets]
    strikes = sorted(list(set([central_strike] + neighbors)))
    return strikes

# Test cases
test_cases = [
    (100.0, 100.0),
    (102.0, 100.0),
    (170.0, 170.0),
    (273.5, 275.0),
    (670.0, 670.0)
]

for spot, central in test_cases:
    strikes = calculate_strikes(spot, central)
    print(f"Spot: {spot}, Central: {central} -> Strikes: {strikes}")
