"""
causal_mask_test.py
-------------------
Thin entry-point kept for convenience.
All test logic lives in:  ../unit tests/test_all.py  (TestCausalMask class)

Run the full suite:
    python -m pytest "../unit tests/test_all.py" -v -k "CausalMask"
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    suite = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "unit tests", "test_all.py",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", suite, "-v", "-k", "CausalMask"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    sys.exit(result.returncode)