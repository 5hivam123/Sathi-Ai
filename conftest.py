"""
Pytest configuration.
Adds the project root to the Python path so tests can import src modules.
"""
import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
