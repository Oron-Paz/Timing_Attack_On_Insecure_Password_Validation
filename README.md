# Timing-Based Password Cracker

This project is a timing-based password cracker designed for educational purposes. It exploits server response time variations to determine the length and content of a password.
It is based on the first assigment of the Attacks on Secure Implementations of Systems course by Yossi Oren

## Files Included:

Main.py: reliable/consistant code that will work most of the time as long as all the paramametrs make sense for the difficulty

Expiremental.py: My expirements with trying to make the code more efficant, if any of my attempts worked I would add them to Main.py

Code.js: The actual code that creates and verifies your password based on your username. Here you can see where the extra noise comes from when the difficutly is increased.

## Features

- **Password Length Detection**: Determines the password length by analyzing response times.
- **Character Discovery**: Identifies characters in the password based on timing differences.
- **Concurrency**: Uses multithreading for faster performance.
- **Statistical Analysis**: Implements a T-test for precise decision-making.

## Requirements

- Python 3.8 or higher
- Required libraries:
  - `requests`
  - `scipy`
  - `python-dotenv`

Install dependencies with:

```bash
pip install requests scipy python-dotenv
```

## Setup

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2. **Create a `.env` file in the project directory and define your user ID:**

    ```bash
    echo "ID=<your_user_id>" > .env
    ```

3. **Update the configuration in the script as needed:**
   - `SERVER_URL`: URL of the server to target.
   - `DIFFICULTY`: Difficulty level for password cracking.
   - `RETRIES`: Number of measurements per character (higher values improve accuracy but increase runtime).
   - `WORKERS`: Number of concurrent threads (adjust to avoid server overload).

## Usage

Run the script to determine the password:

```bash
python Main.py [specify your username in an .env and name it ID]
```

## Steps Performed
**1. Determine Password Length**: The script analyzes response times for passwords of varying lengths to estimate the correct length.
**2. Find Password Characters**: For each position in the password, the script tests all possible characters and selects the one with the highest likelihood based on response times.

## Example Output
```plaintext
Copy
Determining password length...
Length 5, median time: 0.05000s
Estimated length: 5

Searching for password of length 5...
Testing 'a....' at position 0...
Character 'a' median time: 0.07500s
Best character at position 0: 'a'

Final password: abcde
```
## Notes
Ethical Use Only: This tool is for educational and authorized testing purposes. Do not use it for malicious activities. (it also likely wont work for real life scenarios)
Adjust the configuration parameters based on server performance to avoid crashes.

