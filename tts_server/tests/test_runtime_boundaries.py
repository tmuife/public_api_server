from __future__ import annotations

import subprocess
import sys
import unittest


class RuntimeBoundaryTests(unittest.TestCase):
    def test_import_boundaries_script_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check_runtime_import_boundaries.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"Boundary check failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
