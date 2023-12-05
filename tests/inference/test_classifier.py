import subprocess


def test_classifier():
    command = [
        'python',
        'src/classifier.py',
        'This is a test comment for classification.'
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, f"Script execution failed with error: \n{result.stderr}"
    assert "toxic:" in result.stdout
    assert "severe_toxic:" in result.stdout
    assert "obscene:" in result.stdout
    assert "threat:" in result.stdout
    assert "insult:" in result.stdout
    assert "identity_hate:" in result.stdout

