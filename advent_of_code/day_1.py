from typing import List, Tuple

def read_file() -> List[int]:
    file_contents = []
    with open('files/day_1.txt', 'r') as f:
        for line in f:
            file_contents.append(line)

    return file_contents


def find_two_sum_elements(numbers: List[int], sum: int) -> Tuple[int, int]:
    """
    Find two elements in list "numbers" that add to "sum"
    """
    left_pointer = 0
    right_pointer = len(numbers) - 1

    numbers = sorted(numbers)

    while left_pointer != right_pointer:
        large_num, small_num = numbers[left_pointer], numbers[right_pointer]
        current_sum = large_num + small_num
        if current_sum == sum:
            return large_num, small_num
        elif current_sum < sum:
            left_pointer += 1
        else:
            right_pointer -= 1

    return None, None


def find_three_sum_elements(numbers: List[int], sum: int) -> Tuple[int, int, int]:
    """
    Find three elements in list "numbers" that add to "sum"
    """
    numbers = sorted(numbers)

    first_pointer = 0
    while first_pointer < len(numbers):
        second_pointer = first_pointer + 1
        while second_pointer < len(numbers):
            third_pointer = second_pointer + 1
            # Check we're not exceeding sum
            if numbers[first_pointer] + numbers[second_pointer] > sum:
                break

            while third_pointer < len(numbers):
                num_a, num_b, num_c = numbers[first_pointer], numbers[second_pointer], numbers[third_pointer]

                current_sum = num_a + num_b + num_c
                if current_sum == sum:
                    return num_a, num_b, num_c
                elif current_sum > sum:
                    break
                
                third_pointer += 1
            
            second_pointer += 1

        first_pointer += 1
    
    return None, None, None

def main() -> None:
    str_numbers = read_file()
    numbers = [int(line) for line in str_numbers]

    num_a, num_b = find_two_sum_elements(numbers, 2020)

    print(f"First number: {num_a}, Second number: {num_b}, Multiple: {num_a * num_b}")

    num_a, num_b, num_c = find_three_sum_elements(numbers, 2020)
    print(f"First number: {num_a}, Second number: {num_b}, Third number: {num_c}, Multiple: {num_a * num_b * num_c}")

if __name__ == "__main__":
    main()
