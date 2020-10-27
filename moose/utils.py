from pathlib import Path


def load_certificate(filename):
    file = Path(filename) if filename else None
    if file and file.exists():
        with open(str(file), "rb") as f:
            cert = f.read()
            return cert
    return None
