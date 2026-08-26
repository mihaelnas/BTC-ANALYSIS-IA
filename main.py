#!/usr/bin/env python3
"""
LOB Predictor - Main entry point.

This is the easiest way to use the CLI:
  python main.py help              Show help
  python main.py collect           Collect data
  python main.py features          Process features
  python main.py train             Train model
  python main.py evaluate          Evaluate model
  python main.py drift-check       Check data drift
  python main.py serve             Launch web app
"""

from scripts.cli import main

if __name__ == "__main__":
    main()
