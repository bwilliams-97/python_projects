from typing import List
from day_9 import load_numbers
from collections import Counter

def calculate_chain(numbers: List[int]) -> List[int]:
    numbers.append(0)  # Wall joltage
    numbers.append(max(numbers)+3)  # Device joltage
    sorted_nums = sorted(numbers)

    diffs = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]
    diff_counter = Counter(diffs)
    return diff_counter[1] * diff_counter[3]

def calculate_configurations(numbers: List[int]) -> int:
    sorted_nums = sorted(numbers)
    configurations = [None] * len(numbers)

    configurations[0] = 1
    i = 1
    while i < len(numbers):
        configurations[i] = configurations[i-1]
        if i-2 >=0 and (sorted_nums[i] - sorted_nums[i-2] <= 3):
            configurations[i] += configurations[i-2]
        if i-3 >=0 and (sorted_nums[i] - sorted_nums[i-3] <= 3):
            configurations[i] += configurations[i-3]
        i += 1

    return configurations[-1]


def main() -> None:
    with open('files/day_10.txt', 'r') as f:
        numbers = load_numbers(f)

    product = calculate_chain(numbers)
    print(f"Product of one diffs and three diffs: {product}")

    configs = calculate_configurations(numbers)
    print(f"Number unique configurations: {configs}")

    
if __name__ == "__main__":
    main()