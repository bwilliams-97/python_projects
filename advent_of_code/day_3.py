class SnowLevel:
    def __init__(self, level_string: str):
        """
        Tree present when # rather than . character
        """
        self.tree_present = [1 if char == "#" else 0 for char in level_string]

class SnowMap:
    def __init__(self, filepath: str):
        self.levels = []
        self.repeating_length = None

        self.read_file_contents(filepath)

    def read_file_contents(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            for i, level_string in enumerate(f):
                if i == 0:
                    self.repeating_length = len(level_string) - 1
                self.levels.append(SnowLevel(level_string))

    def calculate_num_trees(self, right_step: int = 1, down_step: int = 1) -> None:
        """
        Calculate number trees encountered by moving down_step through levels and
        right_step to the right.
        """
        number_trees = 0

        position = 0
        for i, level in enumerate(self.levels):
            if i % down_step != 0:
                continue
            scaled_position = position % self.repeating_length
            number_trees += level.tree_present[scaled_position]
            position += right_step

        return number_trees


def main() -> None:
    filepath = "files/day_3.txt"
    snow_map = SnowMap(filepath)

    step_options = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]
    multiple = 1

    for option in step_options:
        num_trees = snow_map.calculate_num_trees(option[0], option[1])
        print(f"Option: {option}, Number trees: {num_trees}")
        multiple *= num_trees

    print(f"Total multiple: {multiple}")

if __name__ == "__main__":
    main()
