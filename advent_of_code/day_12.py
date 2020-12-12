from typing import Tuple
from day_5 import read_file
import math

class Ship:
    def __init__(self):
        self.bearing = 0
        self.x_pos = 0
        self.y_pos = 0

    def set_waypoint_location(self, x_pos: int, y_pos: int) -> None:
        self.waypoint_x_pos = x_pos
        self.waypoint_y_pos = y_pos

    def waypoint_distance(self) -> int:
        return math.sqrt(self.waypoint_x_pos ** 2 + self.waypoint_y_pos ** 2)

    def waypoint_angle(self) -> int:
        if self.waypoint_x_pos < 0:
            return math.atan(self.waypoint_y_pos / self.waypoint_x_pos) + math.pi
        return math.atan(self.waypoint_y_pos / self.waypoint_x_pos)

    def update_status_first_iter(self, instruction: str) -> None:
        cmd, value = self.split_instruction(instruction)

        if cmd == "L":
            self.bearing += value
        elif cmd == "R":
            self.bearing -= value
        elif cmd == "N":
            self.y_pos += value
        elif cmd == "S":
            self.y_pos -= value
        elif cmd == "E":
            self.x_pos += value
        elif cmd == "W":
            self.x_pos -= value
        elif cmd == "F":
            self.x_pos += value * math.cos(math.pi * self.bearing / 180)
            self.y_pos += value * math.sin(math.pi * self.bearing / 180)

    def update_status_second_iter(self, instruction) -> None:
        cmd, value = self.split_instruction(instruction)

        if cmd == "L":
            waypoint_x_pos = self.waypoint_distance() * math.cos(self.waypoint_angle() + (math.pi * value / 180))
            waypoint_y_pos = self.waypoint_distance() * math.sin(self.waypoint_angle() + (math.pi * value / 180))
            self.waypoint_x_pos, self.waypoint_y_pos = waypoint_x_pos, waypoint_y_pos
        elif cmd == "R":
            waypoint_x_pos = self.waypoint_distance() * math.cos(self.waypoint_angle() - (math.pi * value / 180))
            waypoint_y_pos = self.waypoint_distance() * math.sin(self.waypoint_angle() - (math.pi * value / 180))
            self.waypoint_x_pos, self.waypoint_y_pos = waypoint_x_pos, waypoint_y_pos
        elif cmd == "N":
            self.waypoint_y_pos += value
        elif cmd == "S":
            self.waypoint_y_pos -= value
        elif cmd == "E":
            self.waypoint_x_pos += value
        elif cmd == "W":
            self.waypoint_x_pos -= value
        elif cmd == "F":
            self.x_pos += value * self.waypoint_x_pos
            self.y_pos += value * self.waypoint_y_pos

    def split_instruction(self, instruction: str) -> Tuple[str, int]:
        cmd = instruction[0]
        value = int(instruction.rstrip()[1:])

        return cmd, value

    def calculate_distance(self) -> int:
        return abs(self.x_pos) + abs(self.y_pos)

def main() -> None:
    lines = read_file("files/day_12.txt")

    ship = Ship()
    for line in lines:
        ship.update_status_first_iter(line)
    distance = ship.calculate_distance()
    print(f"Metropolis distance of ship: {distance}")

    new_ship = Ship()
    new_ship.set_waypoint_location(10, 1)
    for line in lines:
        new_ship.update_status_second_iter(line)
    distance = new_ship.calculate_distance()
    print(f"Metropolis distance of new ship: {distance}")


if __name__ == "__main__":
    main()