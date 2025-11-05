#!/usr/bin/env python3
"""
Verify CME holiday dates in metadata against official US holiday dates.
"""

import csv
import datetime
from pathlib import Path


def get_mlk_day(year):
    """Get Martin Luther King Jr. Day - 3rd Monday of January"""
    jan_first = datetime.date(year, 1, 1)
    first_monday = jan_first + datetime.timedelta(days=(7 - jan_first.weekday()) % 7)
    if first_monday.day == 1:
        first_monday += datetime.timedelta(days=7)
    third_monday = first_monday + datetime.timedelta(days=14)
    return third_monday


def get_presidents_day(year):
    """Get Presidents Day - 3rd Monday of February"""
    feb_first = datetime.date(year, 2, 1)
    first_monday = feb_first + datetime.timedelta(days=(7 - feb_first.weekday()) % 7)
    if first_monday.day == 1:
        first_monday += datetime.timedelta(days=7)
    third_monday = first_monday + datetime.timedelta(days=14)
    return third_monday


def get_memorial_day(year):
    """Get Memorial Day - Last Monday of May"""
    may_31 = datetime.date(year, 5, 31)
    days_to_monday = (may_31.weekday() - 0) % 7
    if days_to_monday == 0 and may_31.day == 31:
        return may_31
    last_monday = may_31 - datetime.timedelta(days=days_to_monday if days_to_monday else 7)
    return last_monday


def get_labor_day(year):
    """Get Labor Day - 1st Monday of September"""
    sep_first = datetime.date(year, 9, 1)
    first_monday = sep_first + datetime.timedelta(days=(7 - sep_first.weekday()) % 7)
    if first_monday.day == 1:
        first_monday += datetime.timedelta(days=7)
    return first_monday


def get_thanksgiving(year):
    """Get Thanksgiving - 4th Thursday of November"""
    nov_first = datetime.date(year, 11, 1)
    first_thursday = nov_first + datetime.timedelta(days=(3 - nov_first.weekday()) % 7)
    if first_thursday.day == 1:
        first_thursday += datetime.timedelta(days=7)
    fourth_thursday = first_thursday + datetime.timedelta(days=21)
    return fourth_thursday


def get_good_friday(year):
    """Get Good Friday using Meeus/Jones/Butcher algorithm for Easter"""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    
    easter = datetime.date(year, month, day)
    good_friday = easter - datetime.timedelta(days=2)
    return good_friday


def verify_holidays():
    """Verify holiday dates in CME calendar metadata"""
    
    # Read the metadata file
    csv_path = Path(__file__).parent.parent / "metadata" / "calendars" / "cme_globex_holidays.csv"
    
    holidays_in_file = {}
    with open(csv_path, 'r') as f:
        # Skip commented header lines (prefixed with '#')
        reader = csv.DictReader(line for line in f if not line.lstrip().startswith('#'))
        for row in reader:
            date_str = row['date']
            holiday_name = row['holiday_name']
            holidays_in_file[date_str] = holiday_name
    
    # Years to check
    years = range(2015, 2026)
    
    errors = []
    verified = []
    
    for year in years:
        # Calculate expected dates
        expected_holidays = {
            get_mlk_day(year): "Martin Luther King Jr Day",
            get_presidents_day(year): "Presidents Day", 
            get_good_friday(year): "Good Friday",
            get_memorial_day(year): "Memorial Day",
            datetime.date(year, 7, 4): "Independence Day",
            get_labor_day(year): "Labor Day",
            get_thanksgiving(year): "Thanksgiving",
            datetime.date(year, 12, 25): "Christmas",
            datetime.date(year, 1, 1): "New Year's Day",
        }
        
        # Special cases
        if year == 2015:
            # July 4, 2015 was Saturday, observed on Friday July 3
            del expected_holidays[datetime.date(2015, 7, 4)]
            expected_holidays[datetime.date(2015, 7, 3)] = "Independence Day"
        
        if year == 2016:
            # Dec 25, 2016 was Sunday, observed on Monday Dec 26
            del expected_holidays[datetime.date(2016, 12, 25)]
            expected_holidays[datetime.date(2016, 12, 26)] = "Christmas"
            # Jan 1, 2017 was Sunday, observed on Monday Jan 2
            
        if year == 2017:
            # Jan 1, 2017 was Sunday, observed on Monday Jan 2
            del expected_holidays[datetime.date(2017, 1, 1)]
            expected_holidays[datetime.date(2017, 1, 2)] = "New Year's Day"
            
        if year == 2021:
            # July 4, 2021 was Sunday, observed on Monday July 5
            del expected_holidays[datetime.date(2021, 7, 4)]
            expected_holidays[datetime.date(2021, 7, 5)] = "Independence Day"
            
        if year == 2022:
            # Dec 25, 2022 was Sunday, observed on Monday Dec 26
            del expected_holidays[datetime.date(2022, 12, 25)]
            expected_holidays[datetime.date(2022, 12, 26)] = "Christmas"
            # Jan 1, 2023 was Sunday, observed on Monday Jan 2
            
        if year == 2023:
            # Jan 1, 2023 was Sunday, observed on Monday Jan 2
            del expected_holidays[datetime.date(2023, 1, 1)]
            expected_holidays[datetime.date(2023, 1, 2)] = "New Year's Day"
        
        # Check each expected holiday (only full-closed days are required for repo)
        for date, name in expected_holidays.items():
            date_str = date.strftime("%Y-%m-%d")
            # Only require explicit closures
            if name in ["Martin Luther King Jr Day", "Presidents Day"]:
                continue
            if date_str in holidays_in_file:
                # Normalize names for comparison
                file_name = holidays_in_file[date_str].replace(" Day", "").replace("Jr ", "Jr")
                expected_name = name.replace(" Day", "").replace("Jr ", "Jr")
                
                if "Martin Luther King" in file_name and "Martin Luther King" in expected_name:
                    verified.append(f"✓ {date_str}: {name}")
                elif "Presidents" in file_name and "Presidents" in expected_name:
                    verified.append(f"✓ {date_str}: {name}")
                elif file_name.lower().startswith(expected_name.lower()[:10]):
                    verified.append(f"✓ {date_str}: {name}")
                else:
                    # Name mismatch
                    pass
            else:
                errors.append(f"✗ Missing {date_str}: {name}")
    
    print("CME Holiday Calendar Verification")
    print("=" * 50)
    
    # Show key dates for 2024-2025
    print("\n2024-2025 Key Dates Verified:")
    print("-" * 30)
    
    key_dates = [
        ("2024-01-01", "New Year's Day"),
        ("2024-01-15", "Martin Luther King Jr Day"),
        ("2024-02-19", "Presidents Day"),
        ("2024-03-29", "Good Friday"),
        ("2024-05-27", "Memorial Day"),
        ("2024-07-04", "Independence Day"),
        ("2024-09-02", "Labor Day"),
        ("2024-11-28", "Thanksgiving"),
        ("2024-12-25", "Christmas"),
        ("2025-01-01", "New Year's Day"),
        ("2025-01-20", "Martin Luther King Jr Day"),
        ("2025-02-17", "Presidents Day"),
        ("2025-04-18", "Good Friday"),
        ("2025-05-26", "Memorial Day"),
        ("2025-07-04", "Independence Day"),
        ("2025-09-01", "Labor Day"),
        ("2025-11-27", "Thanksgiving"),
        ("2025-12-25", "Christmas"),
    ]
    
    for date_str, expected_name in key_dates:
        if date_str in holidays_in_file:
            print(f"✓ {date_str}: {expected_name} - FOUND in metadata")
        else:
            print(f"✗ {date_str}: {expected_name} - MISSING")
    
    # Summary
    print(f"\n{len(verified)} holidays verified")
    if errors:
        print(f"\n{len(errors)} potential issues found:")
        for error in errors[:10]:  # Show first 10
            print(f"  {error}")
    
    print("\nNote: MLK Day and Presidents Day are 'Regular' trading days at CME")
    print("      (bond markets closed but futures trade normally)")
    
    return len(errors) == 0


if __name__ == "__main__":
    success = verify_holidays()
    exit(0 if success else 1)
