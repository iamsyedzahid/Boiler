numbers = (-10, 20, -5, 4, 3, -2, 1)

# Initialize variables
closest_pair = None
min_sum = float('inf')

# Check all pairs
for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):
        pair_sum = numbers[i] + numbers[j]
        if abs(pair_sum) < abs(min_sum):
            min_sum = pair_sum
            closest_pair = (numbers[i], numbers[j])

print(f"The pair of numbers whose sum is closest to zero is: {closest_pair}")
