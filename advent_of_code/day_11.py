from typing import List
from copy import deepcopy

_statuses = {
    "#": 1,
    "L": 0,
    ".": -1
}

class Space:
    def __init__(self, status: int):
        self.status = status

    def update(self):
        if self.status == -1:
            raise ValueError("Floor space cannot be updated")
        self.status = int(not(self.status))

def read_file(f) -> List[List[Space]]:
    grid = []
    for line in f:
        grid.append([Space(_statuses[char]) for char in line.strip()])

    return grid

def find_number_adjacent_occupied(grid: List[List[Space]], row_i: int, col_i: int) -> int:
    return sum([grid[row][col].status == 1 for row in range(max(0, row_i - 1), min(row_i + 2, len(grid))) for col in range(max(0, col_i - 1), min(len(grid[0]), col_i + 2))]) \
        - int(grid[row_i][col_i].status == 1)

def update_grid(grid: List[List[Space]]) -> int:
    new_grid = deepcopy(grid)
    number_updates = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            # Check if floor
            if grid[row][col].status == -1:
                continue
            number_adjacent = find_number_adjacent_occupied(grid, row, col)
            if grid[row][col].status == 0 and number_adjacent == 0:
                new_grid[row][col].update()
                number_updates += 1
            elif grid[row][col].status == 1 and number_adjacent >= 4:
                new_grid[row][col].update()
                number_updates += 1

    return new_grid, number_updates

def total_occupied_seats(grid: List[List[int]]) -> int:
    return sum([space.status == 1 for row in grid for space in row])
            

def main() -> None:
    with open('files/day_11.txt', 'r') as f:
        grid = read_file(f)

    grid, number_updates = update_grid(grid)
    while number_updates > 0:
        grid, number_updates = update_grid(grid)

    print(f"Total occupied seats: {total_occupied_seats(grid)}")

    
    
if __name__ == "__main__":
    main()