from typing import List, Tuple
import re

_bags = {}

class Bag:
    def __init__(self, colour: str):
        self.colour = colour
        _bags[colour] = self

        # Dictionary mapping colour to number of bags
        self.children = {}
        self.contain_shiny_gold = None

    def add_children(self, children: List[Tuple[int, str]]):
        for child_colour in children:
            if child_colour[1] not in _bags:
               _bags[child_colour[1]] = Bag(child_colour[1])
            self.children[child_colour[1]] = child_colour[0]

def parse_line(line: str) -> Tuple[str, List[str]]:
    """
    Read line to get bag contents
    """
    parent_bag_colour = re.match(".+bags contain", line).group()[:-13]
    child_bag_colours = [(int(x.split(' ')[0]), ' '.join(x.split(' ')[1:3])) for x in re.findall("[0-9]\s[a-z]+\s[a-z]+\sbags?[,|.]", line)]
    return parent_bag_colour, child_bag_colours

def check_contain_shiny_gold(bag_colour: Bag) -> bool:
    """
    Recursive function that looks at bag children to see if it can contain a shiny gold bag.
    NOTE: won't work for cyclic graphs...
    """
    # If we haven't seen this one already
    bag = _bags[bag_colour]
    if bag.contain_shiny_gold is None:
        # Check the children
        if bag.children:
            bag.contain_shiny_gold = max([check_contain_shiny_gold(child) for child in bag.children])
        else:
            # If no children then can't contain shiny gold bags
            bag.contain_shiny_gold = False

    # This is the end case (shiny gold bags might not be able to contain others, so can't use default)
    if bag.colour == "shiny gold":
        return True
    
    return bag.contain_shiny_gold

def count_child_bags(bag_colour: str) -> int:
    """
    Recursive function to count number of child bags contained by a particular bag.
    """
    # Each bag contains it's child bags PLUS their children.
    return sum([(count_child_bags(child_bag_colour) + 1) * quantity for child_bag_colour, quantity in _bags[bag_colour].children.items()])


def file_to_bags(f) -> int:
    """
    Go from file to number of bags that can contain shiny gold.
    """
    for line in f:
        parent_bag_colour, child_bag_colours = parse_line(line)
        if parent_bag_colour not in _bags:
            _bags[parent_bag_colour] = Bag(parent_bag_colour)
        _bags[parent_bag_colour].add_children(child_bag_colours)

    for bag_colour, _ in _bags.items():
        check_contain_shiny_gold(bag_colour)

    total_containing_shiny = sum([bag.contain_shiny_gold for bag in _bags.values()])

    total_shiny_contains = count_child_bags("shiny gold")

    return total_containing_shiny, total_shiny_contains
    

def main() -> None:
    with open('files/day_7.txt', 'r') as f:
        a, b = file_to_bags(f)
        print(f"Number bags containing shiny gold: {a}")
        print(f"Number bags shiny contains: {b}")

if __name__ == "__main__":
    main()