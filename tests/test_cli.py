import subprocess
import sys

def test_cli():

    result = subprocess.run(
        [sys.executable, "-m", "qube.cli", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0