import functools

def read_file(filepath: str) -> str:
    with open(filepath, 'r') as f:
        string_contents = f.read()
    return string_contents

def count_unique_characters(string: str) -> int:
    unique_chars = set(string)
    unique_chars.discard('\n')
    return len(unique_chars)

def count_duplicated_characters(string: str) -> int:
    characters = [set(person) for person in string.split('\n')]
    combined = functools.reduce(lambda x1, x2: x1.intersection(x2), characters)

    return len(combined)

def main() -> None:
    contents = read_file("files/day_6.txt")
    groups = contents.split("\n\n")

    total_score = sum([count_unique_characters(group) for group in groups])
    print(f"Total score: {total_score}")

    total_score_intersection = sum([count_duplicated_characters(group) for group in groups])
    print(f"Total score intersection: {total_score_intersection}")



if __name__ == "__main__":
    main()