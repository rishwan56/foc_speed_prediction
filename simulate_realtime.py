import os
import sys
import time
import random
import argparse
from datetime import datetime
import pandas as pd

from predict import predict_speed

# Realistic operational boundaries based on dataset distributions
BOUNDS = {
    'Temperature_C': (22.0, 40.0),
    'Humidity_percent': (30.0, 90.0),
    'Gas_ppm': (50.0, 350.0),
    'Pressure_hPa': (995.0, 1025.0),
    'Occupancy': (0, 10),
    'Voltage_V': (23.0, 25.2),
    'Current_A': (1.0, 5.2)
}

# Real-world mall scenarios for testing
SCENARIOS = {
    'morning_empty': {
        'desc': 'Mall Opening / Low Traffic (Morning)',
        'temp_range': (22.0, 25.0),
        'humidity_range': (40.0, 55.0),
        'gas_range': (50.0, 90.0),
        'occupancy_range': (0, 2),
    },
    'afternoon_crowd': {
        'desc': 'Peak Afternoon Weekend Rush',
        'temp_range': (33.0, 38.0),
        'humidity_range': (65.0, 85.0),
        'gas_range': (220.0, 330.0),
        'occupancy_range': (7, 10),
    },
    'normal_evening': {
        'desc': 'Moderate Evening Mall Activity',
        'temp_range': (28.0, 32.0),
        'humidity_range': (50.0, 65.0),
        'gas_range': (150.0, 220.0),
        'occupancy_range': (4, 6),
    },
    'food_court_smoke': {
        'desc': 'Cooking/Smoke spike in Food Court',
        'temp_range': (31.0, 36.0),
        'humidity_range': (60.0, 80.0),
        'gas_range': (290.0, 350.0),
        'occupancy_range': (6, 9),
    }
}


def generate_random_sample(scenario=None, previous_state=None):
    """
    Generates realistic environmental sensor readings.
    If previous_state is provided, it simulates smooth real-world sensor drift (random walk).
    """
    if scenario and scenario in SCENARIOS:
        cfg = SCENARIOS[scenario]
        temp = random.uniform(*cfg['temp_range'])
        humidity = random.uniform(*cfg['humidity_range'])
        gas = random.uniform(*cfg['gas_range'])
        occupancy = random.randint(*cfg['occupancy_range'])
    elif previous_state:
        # Realistic drift: smooth fluctuation from previous state
        temp = np_clip(previous_state['Temperature_C'] + random.uniform(-0.4, 0.4), *BOUNDS['Temperature_C'])
        humidity = np_clip(previous_state['Humidity_percent'] + random.uniform(-1.0, 1.0), *BOUNDS['Humidity_percent'])
        gas = np_clip(previous_state['Gas_ppm'] + random.uniform(-8.0, 8.0), *BOUNDS['Gas_ppm'])
        occ_delta = random.choice([-1, 0, 0, 1])
        occupancy = int(np_clip(previous_state['Occupancy'] + occ_delta, *BOUNDS['Occupancy']))
    else:
        temp = random.uniform(*BOUNDS['Temperature_C'])
        humidity = random.uniform(*BOUNDS['Humidity_percent'])
        gas = random.uniform(*BOUNDS['Gas_ppm'])
        occupancy = random.randint(*BOUNDS['Occupancy'])

    pressure = random.uniform(1008.0, 1015.0)
    # Voltage typically nominal 24V with minor fluctuations
    voltage = round(random.uniform(23.8, 24.4), 2)
    # Current correlates with occupancy/temp load
    current = round(np_clip(1.2 + (occupancy * 0.28) + ((temp - 22.0) * 0.08) + random.uniform(-0.15, 0.15), *BOUNDS['Current_A']), 2)

    return {
        'Temperature_C': round(temp, 1),
        'Humidity_percent': round(humidity, 1),
        'Gas_ppm': round(gas, 1),
        'Pressure_hPa': round(pressure, 2),
        'Occupancy': float(occupancy),
        'Voltage_V': voltage,
        'Current_A': current
    }


def np_clip(val, min_v, max_v):
    return max(min_v, min(max_v, val))


def render_speed_bar(rpm, min_rpm=900, max_rpm=3000, width=20):
    clamped_rpm = max(min_rpm, min(max_rpm, rpm))
    pct = (clamped_rpm - min_rpm) / (max_rpm - min_rpm)
    filled = int(round(pct * width))
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {rpm:7.1f} RPM ({int(pct * 100)}%)"


def run_continuous_simulation(interval=1.0, steps=None, scenario=None, log_file=None):
    """
    Runs an interactive continuous sensor loop to test real-time FOC speed prediction.
    """
    print("\n" + "=" * 90)
    print(" FOC BLDC MOTOR SPEED PREDICTION - REAL-TIME SENSOR SIMULATOR")
    print("=" * 90)
    print(f"Update Interval: {interval}s | Max Steps: {'Unlimited' if steps is None else steps}")
    if scenario:
        print(f"Scenario Preset: {scenario} ({SCENARIOS[scenario]['desc']})")
    print("Press Ctrl+C at any time to pause or exit.\n")

    # Table Header
    header = (
        f"{'Step':<5} | {'Time':<8} | {'Temp(C)':<8} | {'Hum(%)':<6} | "
        f"{'Gas(ppm)':<8} | {'People':<6} | {'V(V)':<5} | {'I(A)':<5} | {'FOC Predicted Speed / Motor Drive'}"
    )
    print(header)
    print("-" * 90)

    history = []
    current_state = None
    step_count = 0

    try:
        while True:
            step_count += 1
            current_state = generate_random_sample(scenario=scenario, previous_state=current_state)
            
            # Predict speed using trained KNN model
            predicted_rpm = predict_speed(current_state)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            speed_bar = render_speed_bar(predicted_rpm)

            row = (
                f"{step_count:<5} | {timestamp:<8} | "
                f"{current_state['Temperature_C']:<8.1f} | "
                f"{current_state['Humidity_percent']:<6.1f} | "
                f"{current_state['Gas_ppm']:<8.1f} | "
                f"{int(current_state['Occupancy']):<6} | "
                f"{current_state['Voltage_V']:<5.1f} | "
                f"{current_state['Current_A']:<5.2f} | "
                f"{speed_bar}"
            )
            print(row)

            # Record for logging
            record = current_state.copy()
            record['Step'] = step_count
            record['Timestamp'] = timestamp
            record['Predicted_RPM'] = round(predicted_rpm, 2)
            history.append(record)

            if steps is not None and step_count >= steps:
                print("\nReached target simulation steps.")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n[Simulation paused by user]")

    if log_file:
        df = pd.DataFrame(history)
        df.to_csv(log_file, index=False)
        print(f"Simulation logs saved to {log_file}")

    print("=" * 92 + "\n")


def run_batch_test(count=5):
    """
    Generates a table of distinct random test cases to verify model stability.
    """
    print(f"\n--- Generating {count} Random Test Scenarios ---")
    results = []
    for i in range(1, count + 1):
        sample = generate_random_sample()
        rpm = predict_speed(sample)
        sample['Predicted_RPM'] = round(rpm, 1)
        results.append(sample)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("--------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Mall Environment Simulator for FOC BLDC Drive")
    parser.add_argument('--interval', type=float, default=1.0, help="Seconds between sensor updates (default: 1.0)")
    parser.add_argument('--steps', type=int, default=None, help="Number of steps to simulate (default: continuous)")
    parser.add_argument('--scenario', choices=list(SCENARIOS.keys()), default=None,
                        help="Pre-configured mall condition: morning_empty, afternoon_crowd, normal_evening, food_court_smoke")
    parser.add_argument('--batch', type=int, default=None, help="Run a quick batch test of N random test cases")
    parser.add_argument('--log', type=str, default=None, help="Save simulation output to CSV file (e.g. sim_output.csv)")

    args = parser.parse_args()

    if args.batch:
        run_batch_test(count=args.batch)
    else:
        run_continuous_simulation(
            interval=args.interval,
            steps=args.steps,
            scenario=args.scenario,
            log_file=args.log
        )


if __name__ == '__main__':
    main()
