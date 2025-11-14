# FootballP Tracker

## Installation

1. **Install [uv](https://github.com/astral-sh/uv):**

    Using pip:
    ```bash
    pip install uv
    ```

    Or using curl (Linux/macOS):
    ```bash
    curl -Ls https://astral.sh/uv/install.sh | bash
    ```

2. **Sync dependencies using `uv.lock`:**
    ```bash
    uv pip sync
    ```
    This will install all dependencies listed in your `uv.lock` file.

## Running the Application

To run `main.py` using uv:
```bash
uv run main.py
```