from typing import List
from day_1 import find_two_sum_elements

def load_numbers(f) -> List[int]:
    numbers = []
    for line in f:
        numbers.append(int(line.strip()))

    return numbers

def check_addition_to_x(nums: List[int], x: int) -> bool:
    """
    Check if there are two numbers in nums that add to x.
    """
    a, b = find_two_sum_elements(nums, x)
    if a is None or b is None:
        return False
    return True

def find_first_invalid_entry(numbers: List[int]) -> int:
    """
    Find first number that can't be expressed as a sum of two from previous 25.
    """
    i = 25
    while i < len(numbers):
        if not check_addition_to_x(numbers[i-25:i], numbers[i]):
            return numbers[i]
        i += 1
    return None

def find_continguous_list(numbers: List[int], total: int) -> List[int]:
    left_pointer = 0
    right_pointer = 1
    list_sum = 0
    while list_sum != total:
        if list_sum < total:
            right_pointer += 1
        if list_sum > total:
            left_pointer += 1
        list_sum = sum(numbers[left_pointer: right_pointer])

    return numbers[left_pointer: right_pointer]


def main() -> None:
    with open('files/day_9.txt', 'r') as f:
        numbers = load_numbers(f)

    first_invalid_entry = find_first_invalid_entry(numbers)
    print(f"First invalid entry: {first_invalid_entry}")

    continguous_list = find_continguous_list(numbers, first_invalid_entry)
    print(f"Sum of smallest and largest: {min(continguous_list) + max(continguous_list)}")

if __name__ == "__main__":
    main()
