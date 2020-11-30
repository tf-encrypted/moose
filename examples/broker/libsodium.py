import numpy as np
import pysodium

if __name__ == "__main__":
    # m = b"howdy"
    # m = bytes(3)
    a = np.array([1, 2, 3, 4])
    m = a.tobytes()
    pk, sk = pysodium.crypto_box_keypair()

    # n = pysodium.randombytes(pysodium.crypto_box_NONCEBYTES)
    n = (1).to_bytes(24, "little")
    c = pysodium.crypto_box(m, n, pk, sk)
    plaintext = pysodium.crypto_box_open(c, n, pk, sk)
    print("Test")
    assert m == plaintext
