"""
Pytest conftest for DeepEval framework.
Configures DeepEval and provides shared fixtures.
"""
import os
import sys

# Ensure the framework directory is in path
sys.path.insert(0, os.path.dirname(__file__))
