def simple_probability(m, n):
    if not isinstance(m, int) or not isinstance(n, int):
        raise TypeError("m and n must be integers")
    if n <= 0:
        raise ValueError("n must be > 0")
    if m < 0 or m > n:
        raise ValueError("m must be between 0 and n (inclusive)")
    return m / n

def logical_or(m, k, n):
    if not isinstance(m, int) or not isinstance(k, int) or not isinstance(n, int):
        raise TypeError("m, k and n must be integers")
    if n <= 0:
        raise ValueError("n must be > 0")
    if m < 0 or k < 0:
        raise ValueError("m and k must be >= 0")
    if m + k > n:
        raise ValueError("m + k must be <= n")
    return (m + k) / n

def logical_and(m, k, n, l):
    if not all(isinstance(x, int) for x in (m, k, n, l)):
        raise TypeError("m, k, n and l must be integers")
    if n <= 0 or l <= 0:
        raise ValueError("n and l must be > 0")
    if m < 0 or k < 0:
        raise ValueError("m and k must be >= 0")
    if m > n or k > l:
        raise ValueError("m must be <= n and k must be <= l")
    return (m / n) * (k / l)

def expected_value(values, probabilities):
    if not (isinstance(values, tuple) and isinstance(probabilities, tuple)):
        raise TypeError("values and probabilities must be tuples")
    if len(values) == 0 or len(values) != len(probabilities):
        raise ValueError("values and probabilities must be non-empty tuples of the same length")
    if any(p < 0 for p in probabilities):
        raise ValueError("probabilities must be non-negative")
    total_prob = sum(probabilities)
    if not abs(total_prob - 1.0) <= 1e-8:
        raise ValueError("sum of probabilities must be 1")
    return sum(v * p for v, p in zip(values, probabilities))

def conditional_probability(values):
    if not isinstance(values, tuple):
        raise TypeError("values must be a tuple of pairs")
    if len(values) == 0:
        raise ValueError("values must be non-empty")

    count_first_one = 0
    count_both_one = 0

    for pair in values:
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise ValueError("each item in values must be a tuple of length 2")

        a, b = pair
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("pair elements must be integers 0 or 1")
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError("pair elements must be 0 or 1")

        if a == 1:
            count_first_one += 1
            if b == 1:
                count_both_one += 1

    if count_first_one == 0:
        raise ValueError(
            "no occurrences where first event equals 1; "
            "conditional probability undefined"
        )

    return count_both_one / count_first_one

def bayesian_probability(a, ba):
    if not (0.0 <= a <= 1.0) or not (0.0 <= ba <= 1.0):
        raise ValueError("a и ba должны быть в диапазоне [0, 1].")
    if ba == 0.0:
        return 0.0
    return (ba * a) / b


if __name__ == "__main__":
    print("Simple Probability (3 of 10):", simple_probability(3, 10))
    print("Logical OR (3 or 2 of 10):", logical_or(3, 2, 10))
    print("Logical AND (3 of 10 and 2 of 8):", logical_and(3, 2, 10, 8))
    print("Expected Value:", expected_value((1, 2, 3), (0.2, 0.5, 0.3)))
    sample_pairs = ((1, 1), (1, 0), (0, 1), (1, 1))
    print("Conditional Probability:", conditional_probability(sample_pairs))
    print("Bayesian Probability:", bayesian_probability(0.1, 0.8))