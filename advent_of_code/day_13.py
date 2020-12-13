from typing import Tuple, List

def read_file(f) -> Tuple[int, List[int]]:
    contents = (line for line in f)
    earliest_time = int(next(contents))

    bus_time_contents = next(contents)
    bus_times = []
    for i, time in enumerate(bus_time_contents.split(',')):
        if time != "x":
            # Tuple of bus time and position in list
            bus_times.append((int(time), i))

    return earliest_time, bus_times

def find_best_bus_time(earliest_time: int, bus_times: List[int]) -> Tuple[int, int]:
    """
    Find bus that has best time, and time we would have to wait.
    """
    def calculate_wait(time_available: int, bus_interval: int) -> int:
        return bus_interval - (time_available % bus_interval)

    wait_times = [(bus_time, calculate_wait(earliest_time, bus_time)) for bus_time in bus_times]

    return min(wait_times, key=lambda x: x[1])

def find_smallest_matching_position(bus_times: List[Tuple[int, int]]) -> int:
    longest_interval = max(bus_times, key = lambda x: x[0])[0]
    bus_positions = dict(bus_times)

    bus_offsets = {
        bus_time: bus_positions[bus_time] - bus_positions[longest_interval]
        for bus_time in bus_positions.keys()
    }

    i = 1
    match = False
    while not match:
        print(i, end='\r')
        central_position = i * longest_interval
        matched = [(central_position - bus_offsets[bus_time]) % bus_time == 0 for bus_time in bus_positions.keys()]
        match = all(matched)
        i += 1
    return central_position + bus_offsets[bus_times[0][0]]

def main() -> None:
    with open("files/day_13.txt", "r") as f:
        earliest_time, bus_times = read_file(f)

    best_time = find_best_bus_time(earliest_time, [x[0] for x in bus_times])

    print(f"Best bus to get: {best_time[0]}, Wait time: {best_time[1]}")
    print(f"Multiple: {best_time[0] * best_time[1]}")

    earliest_time_to_start = find_smallest_matching_position(bus_times)
    print(f"Shortest waiting time: {earliest_time_to_start}")
    


if __name__ == "__main__":
    main()