# test_notebooks.py

This script is designed to automatically execute Jupyter notebooks in the `notebooks/` directory (or a specified directory) and report their execution status. It is useful for validating that all notebooks in the repository run successfully from start to finish, which is especially helpful for CI/CD pipelines or for contributors to verify their changes.

## Features
- **Automatic Discovery:** Recursively scans a directory for `.ipynb` files (excluding hidden files).
- **Selective Skipping:** Supports a skip list to exclude specific notebooks from execution (e.g., those requiring manual input).
- **Execution Reporting:** Prints a summary of successful and failed notebooks, including error messages for failures.
- **Command Line Usage:** Can run all notebooks in a directory or a specified list of notebook files.

## Usage

### Run All Notebooks in a Directory

```bash
python3 tools/test_notebooks.py
```
This will scan the `notebooks/` directory by default, skipping any notebooks listed in the `skip_list` variable.

### Run Specific Notebooks

```bash
python3 tools/test_notebooks.py notebooks/example1.ipynb notebooks/example2.ipynb
```
This will execute only the specified notebooks.

## Skip List
You can modify the `skip_list` variable in the script to add or remove notebooks that should be skipped during execution. The skip list can contain full paths or substrings.

## Dependencies
- Python 3
- `nbformat`
- `nbconvert`

Install dependencies with:
```bash
pip3 install nbformat nbconvert
```

## Exit Codes
- Returns `0` if all notebooks succeed.
- Returns `1` if any notebook fails or if no notebooks are found.

## Example Output
```
🔍 Scanning for notebooks in: /path/to/notebooks
▶️ Running: /path/to/notebooks/example.ipynb
✅ Success: /path/to/notebooks/example.ipynb
...
🧾 Notebook Execution Summary
✅ 3 succeeded
❌ 1 failed
🚨 Failed notebooks:
 - /path/to/notebooks/failing.ipynb
   ↳ Traceback (most recent call last): ...
```

## Notes
- Notebooks that require manual input or special setup should be added to the skip list.
- The script is intended for use in development, CI, or pre-merge validation workflows.
