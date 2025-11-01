"""Main module entry point."""

# Initialize logging before any other imports
from .utils.logging_setup import setup_logging
setup_logging()

from .main import main

if __name__ == "__main__":
    exit(main())