from typing import List, Tuple
from copy import deepcopy

class Instruction:
    def __init__(self, cmd: str, arg: int):
        self.cmd = cmd
        self.arg = arg

def parse_lines(f) -> List[Instruction]:
    instructions = []
    for line in f:
        split = line.strip().split(' ')
        instruction = Instruction(split[0], int(split[1]))
        instructions.append(instruction)

    return instructions

def follow_instructions(instructions: List[Instruction]) -> Tuple[int, bool]:
    i = 0
    acc = 0
    visited_positions = set()
    while i < len(instructions) and i not in visited_positions:
        visited_positions.add(i)
        cur_op = instructions[i]
        if cur_op.cmd == "nop":
            i +=1
        elif cur_op.cmd == "jmp":
            i += cur_op.arg
        elif cur_op.cmd == "acc":
            acc += cur_op.arg
            i += 1
    reached_end = True if len(instructions) - 1 in visited_positions else False

    return acc, reached_end

def test_for_switch(instructions: List[Instruction]) -> int:
    for i in range(len(instructions)):
        print(i, end='\r')
        if instructions[i].cmd == "nop" or instructions[i].cmd == "jmp":
            ic = deepcopy(instructions)
            ic[i].cmd = "jmp" if instructions[i].cmd == "nop" else "nop"
            acc, reached_end = follow_instructions(ic)
            if reached_end:
                return acc
    return None


def main() -> None:
    with open('files/day_8.txt', 'r') as f:
        instructions = parse_lines(f)
    
    acc, _ = follow_instructions(instructions)

    print(f"Value of accumulator: {acc}")

    acc_if_switched = test_for_switch(instructions)
    print(f"Value of accumulator with switch: {acc_if_switched}")

if __name__ == "__main__":
    main()