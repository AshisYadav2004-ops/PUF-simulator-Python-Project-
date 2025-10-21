
import numpy as np
from typing import Tuple, List


def random_challenges(n: int, k: int, seed: int | None = None) -> np.ndarray:
    
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n, k), dtype=np.int8)


def _challenge_to_phi(challenge: np.ndarray) -> np.ndarray:
    
    k = len(challenge)
    s = 1 - 2 * challenge  
    phi = np.empty(k + 1, dtype=np.int8)
    phi[0] = 1
    
    cum = 1
    for i in range(1, k + 1):
        cum *= s[k - i]  
        phi[i] = cum
    return phi


def challenges_to_phi(challenges: np.ndarray) -> np.ndarray:
    
    n, k = challenges.shape
    phi = np.empty((n, k + 1), dtype=np.int8)
    for i in range(n):
        phi[i] = _challenge_to_phi(challenges[i])
    return phi


class ArbiterPUF:
    
    def __init__(self, k: int, weight_scale: float = 1.0, seed: int | None = None):
        self.k = k
        self.dim = k + 1
        self.rng = np.random.default_rng(seed)
        self.w = self.rng.normal(loc=0.0, scale=weight_scale, size=(self.dim,))

    def copy_with_noise(self, noise_std: float) -> "ArbiterPUF":
        
        clone = ArbiterPUF(self.k)
        clone.w = self.w + self.rng.normal(scale=noise_std, size=self.w.shape)
        return clone

    def delay(self, phi: np.ndarray, noise_std: float = 0.0) -> float:
        
        noise = self.rng.normal(scale=noise_std)
        return float(np.dot(self.w, phi) + noise)

    def response(self, challenge: np.ndarray, noise_std: float = 0.0) -> int:
        
        phi = _challenge_to_phi(challenge)
        d = self.delay(phi, noise_std=noise_std)
        return 1 if d >= 0 else 0

    def responses(self, challenges: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        
        phi = challenges_to_phi(challenges)
        d = phi @ self.w
        if noise_std > 0:
            d = d + self.rng.normal(scale=noise_std, size=d.shape)
        return (d >= 0).astype(np.int8)


class XORArbiterPUF:

    def __init__(self, m: int, k: int, weight_scale: float = 1.0, seed: int | None = None):
        self.m = m
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.components: List[ArbiterPUF] = [
            ArbiterPUF(k, weight_scale=weight_scale, seed=self.rng.integers(0, 2**31 - 1))
            for _ in range(m)
        ]

    def response(self, challenge: np.ndarray, noise_std: float = 0.0) -> int:
        bits = [comp.response(challenge, noise_std=noise_std) for comp in self.components]
        
        return int(np.bitwise_xor.reduce(np.array(bits, dtype=np.int8)))

    def responses(self, challenges: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        comp_resps = np.vstack([comp.responses(challenges, noise_std=noise_std) for comp in self.components])
        xor_all = np.bitwise_xor.reduce(comp_resps, axis=0)
        return xor_all.astype(np.int8)



def generate_dataset(puf, n: int, k: int, seed: int | None = None, noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    
    challenges = random_challenges(n, k, seed=seed)
    responses = puf.responses(challenges, noise_std=noise_std)
    return challenges, responses



def _demo():
    print("=== APUF simulator demo ===")
    k = 16
    n = 1000
    seed = 12345

    apuf = ArbiterPUF(k=k, weight_scale=1.0, seed=seed)
    ch, r = generate_dataset(apuf, n=n, k=k, seed=seed + 1, noise_std=0.0)
    print(f"APUF: generated {n} CRPs, {np.mean(r):.3f} fraction of ones")

    
    _, r_noisy = generate_dataset(apuf, n=n, k=k, seed=seed + 2, noise_std=0.2)
    flips = np.mean(r != r_noisy)
    print(f"APUF: noise-induced response flip fraction (noise_std=0.2): {flips:.3f}")


    xor_puf = XORArbiterPUF(m=3, k=k, weight_scale=1.0, seed=seed)
    ch2, r2 = generate_dataset(xor_puf, n=n, k=k, seed=seed + 3, noise_std=0.0)
    print(f"XOR-APUF (m=3): fraction ones {np.mean(r2):.3f}")

    
    try:
        from sklearn.linear_model import LogisticRegression
        phi = challenges_to_phi(ch)  # shape n x (k+1)
        clf = LogisticRegression(max_iter=500).fit(phi, r)
        acc = clf.score(phi, r)
        print(f"APUF: logistic regression fit accuracy (train): {acc:.3f}")
    except Exception:
        print("sklearn not available — skipping linear separability demo")

    print("Demo finished.")


if __name__ == "__main__":
    _demo()
