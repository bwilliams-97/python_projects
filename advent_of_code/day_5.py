from typing import List, Tuple
import math

ROWS = 128
COLS = 8

def read_file(filepath: str) -> List[str]:
    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            lines.append(line)

    return lines

def get_seat_id(row: int, column: int) -> int:
    return row * 8 + column

def get_seat_position(spec: str) -> Tuple[int, int]:
    min_row = min_col = 0
    max_row = ROWS - 1
    max_col = COLS - 1

    # Get row
    for char in spec[:7]:
        if char == 'B': 
            min_row = math.ceil((max_row + min_row) / 2)
        else:
            max_row = int((max_row + min_row) / 2)
    # Get col
    for char in spec[7:].strip():
        if char == 'R':
            min_col = math.ceil((max_col + min_col) / 2)
        else:
            max_col = int((max_col + min_col) / 2)

    return min_row, min_col

def get_missing_seat_id(all_seat_ids: List[int]) -> int:
    sorted_ids = sorted(all_seat_ids)
    for i in range(len(sorted_ids)):
        if sorted_ids[i + 1] - sorted_ids[i] != 1:
            return sorted_ids[i + 1] - 1
    return None

def main() -> None:
    lines = read_file("files/day_5.txt")
    all_seat_ids = [get_seat_id(*get_seat_position(spec)) for spec in lines]
    breakpoint()
    print(f"Max seat ID: {max(all_seat_ids)}")

    print(f"Missing seat ID: {get_missing_seat_id(all_seat_ids)}")

if __name__ == "__main__":
    main()    