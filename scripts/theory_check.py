"""Check the T1 binary mean model using independent binomial sufficient statistics."""

import csv
import math
import sys

import numpy as np


TRIALS = 100_000
LOWER, UPPER = 0.05, 0.95
CASES = [
    (200, 4, 0.0, 0.6),
    (200, 10, 0.0, 0.6),
    (200, 50, 0.0, 0.6),
    (200, 200, 0.0, 0.6),
    (40, 10, 0.7, 0.2),
    (40, 20, 0.7, 0.2),
    (40, 40, 0.7, 0.2),
    (10_000, 100, 0.0, 0.6),
    (10_000, 100, 0.5, 0.6),
]


def conditional_mse(m, l, mu, b, weight):
    denominator = m + l * weight**2
    variance = m * (1 - mu**2) + l * weight**2 * (1 - (b * mu)**2)
    bias = mu * l * weight * (b - weight)
    return (variance + bias**2) / denominator**2


def exact_mse(m, l, mu, b):
    probability = (1 + b) / 2
    terms = []
    for k in range(m + 1):
        weight = min(UPPER, max(LOWER, 2 * k / m - 1))
        mass = math.comb(m, k) * probability**k * (1 - probability)**(m - k)
        terms.append(mass * conditional_mse(m, l, mu, b, weight))
    return math.fsum(terms)


def simulate(rng, m, l, mu, b):
    # In the model the label, flip, and noisy-label sums are mutually independent.
    k = rng.binomial(m, (1 + b) / 2, TRIALS)
    weight = np.clip(2 * k / m - 1, LOWER, UPPER)
    clean_sum = 2 * rng.binomial(m, (1 + mu) / 2, TRIALS) - m
    noisy_sum = 2 * rng.binomial(l, (1 + b * mu) / 2, TRIALS) - l
    squared_error = ((clean_sum + weight * noisy_sum) / (m + l * weight**2) - mu)**2
    return squared_error.mean(), squared_error.std(ddof=1) / math.sqrt(TRIALS)


def main():
    rng = np.random.default_rng(20261123)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("n", "m", "mu", "b", "trials", "mse_exact", "mse_mc", "mc_se",
                     "z_score", "r_oracle", "r_exact", "r_mc"))
    for n, m, mu, b in CASES:
        l = n - m
        exact = exact_mse(m, l, mu, b)
        estimate, se = simulate(rng, m, l, mu, b)
        z_score = (estimate - exact) / se
        baseline = (1 - mu**2) / m
        oracle = baseline / conditional_mse(m, l, mu, b, b)
        row = (exact, estimate, se, z_score, oracle, baseline / exact, baseline / estimate)
        writer.writerow((n, m, mu, b, TRIALS, *(f"{value:.10g}" for value in row)))
        if abs(z_score) > 5:
            raise RuntimeError(f"MSE differs by more than five MC SEs for {(n, m, mu, b)}")


if __name__ == "__main__":
    main()
