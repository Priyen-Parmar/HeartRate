#!/usr/local/bin/python3.12
"""
Wrapper entry point for backward compatibility.
Delegates directly to the unified VitalEye live camera heart rate monitor (`live_hr_monitor.py`).
"""

from live_hr_monitor import run, parse_args

if __name__ == "__main__":
    run(parse_args())
