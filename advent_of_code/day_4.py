from typing import List
import re

_required_fields = ['byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid']
"""
byr (Birth Year) - four digits; at least 1920 and at most 2002.
iyr (Issue Year) - four digits; at least 2010 and at most 2020.
eyr (Expiration Year) - four digits; at least 2020 and at most 2030.
hgt (Height) - a number followed by either cm or in:
If cm, the number must be at least 150 and at most 193.
If in, the number must be at least 59 and at most 76.
hcl (Hair Color) - a # followed by exactly six characters 0-9 or a-f.
ecl (Eye Color) - exactly one of: amb blu brn gry grn hzl oth.
pid (Passport ID) - a nine-digit number, including leading zeroes.
"""

def read_file(filepath: str) -> str:
    with open(filepath, 'r') as f:
        string_contents = f.read()
    return string_contents

def check_byr(string: str) -> bool:
    birth_year = re.findall("byr\:[0-9]{4}", string)
    if birth_year:
        by = int(birth_year[0][4:])
        
        return by >= 1920 and by <= 2002
    return False

def check_iyr(string: str) -> bool:
    issue_year = re.findall("iyr\:[0-9]{4}", string)
    if issue_year:
        iy = int(issue_year[0][4:])
        
        return iy >= 2010 and iy <= 2020
    return False

def check_eyr(string: str) -> bool:
    exp_year = re.findall("eyr\:[0-9]{4}", string)
    if exp_year:
        ey = int(exp_year[0][4:])
        
        return ey >= 2020 and ey <= 2030
    return False

def check_hgt(string: str) -> bool:
    height = re.findall("hgt\:([0-9]+)(in|cm)", string)
    if height:
        h = height[0][0]
        unit = height[0][1]
        
        return ("cm" in unit and int(h) >=150 and int(h) <=193) or \
            ("in" in unit and int(h) >= 59 and int(h) <= 76)
    return False

def check_hcl(string: str) -> bool:
    hair_col = re.findall("\#[0-9a-f]{6}", string)
    
    return len(hair_col) > 0

def check_ecl(string: str) -> bool:
    ecl = re.findall("ecl\:[a-z]+", string)
    if ecl:
        ecl = ecl[0][4:]
        
        return ecl in ["amb", "blu", "brn", "gry", "grn", "hzl", "oth"]
    return False

def check_pid(string: str) -> bool:
    pid = re.findall("pid\:[0-9]+", string)
    if pid:
        pid = pid[0][4:]
        return len(pid) == 9
    return False

def split_string_by_newlines(string: str) -> List[str]:
    return string.split('\n\n')

def check_field_in_string(field: str, string: str) -> bool:
    return field + ':' in string

def check_passport_valid(passport: str) -> bool:
    return min([check_field_in_string(field, passport) for field in _required_fields])

def check_passport_valid_stringent(passport: str) -> bool:
    return check_passport_valid(passport) and check_byr(passport) and check_ecl(passport) and check_eyr(passport) \
        and check_hcl(passport) and check_hgt(passport) and check_iyr(passport) and check_pid(passport)

def main() -> None:
    filepath = "files/day_4.txt"

    file_contents = read_file(filepath)

    passports = split_string_by_newlines(file_contents)
    
    n_valid_passports = len([p for p in passports if check_passport_valid(p)])
    print(f"Number valid passports: {n_valid_passports}")

    n_valid_passports_2 = len([p for p in passports if check_passport_valid_stringent(p)])
    print(f"Number valid passports stringent: {n_valid_passports_2}")



if __name__ == "__main__":
    main()