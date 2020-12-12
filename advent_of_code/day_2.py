import re

class LineBreakdown:
    def __init__(self, min: int, max: int, character: str, password: str):
        self.min = min
        self.max = max
        self.character = character
        self.password = password

def parse_line(line: str) -> LineBreakdown:
    """
    Convert line in standard form into LineBreakdown.
    E.g. 1-3 x: xypc => Line(1, 3, 'x', 'xypc')
    """
    min = re.findall("[0-9]+\-", line)[0][:-1]
    max = re.findall("\-[0-9]+", line)[0][1:]
    character = re.findall("[a-z]\:", line)[0][:-1]
    password = re.findall("\: [a-z]+", line)[0][2:]
    
    return LineBreakdown(int(min), int(max), character, password)


def count_passwords(f) -> None:
    """
    Count number of valid passwords in given text file.
    Uses two iterations of valid condition.
    """
    total_valid_passwords_first = 0
    total_valid_passwords_second = 0
    
    for line in f:
        line_breakdown = parse_line(line)
        character_count = line_breakdown.password.count(line_breakdown.character)

        # In first iteration, number of occurences of character must be between min and max
        if character_count >= line_breakdown.min and character_count <= line_breakdown.max:
            total_valid_passwords_first += 1

        # In second iteration, character must be in position min OR position max, but not both.
        # I.e. we use an XOR condition. Note that min and max are not zero indexed.
        if ((line_breakdown.password[line_breakdown.min - 1] == line_breakdown.character) != \
            (line_breakdown.password[line_breakdown.max - 1] == line_breakdown.character)):
            total_valid_passwords_second += 1

    print(f"Total valid passwords (iter 1): {total_valid_passwords_first}")
    print(f"Total valid passwords (iter 2): {total_valid_passwords_second}")


def main() -> None:
    with open("files/day_2.txt", 'r') as f:
        count_passwords(f)

if __name__ == "__main__":
    main()