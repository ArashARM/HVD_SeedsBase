import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pred_NN_Classes import PPNet


def main():
    model = PPNet(n_seeds=5)
    uv_init = torch.rand(5, 2)
    out = model(uv_init)

    expected_keys = {
        "seeds_raw",
        "w_raw",
        "h_raw",
        "theta",
        "a_raw",
        "boundary_width_raw",
        "boundary_alpha_raw",
        "boundary_beta_raw",
        "tau",
    }
    assert set(out.keys()) == expected_keys
    assert out["seeds_raw"].shape == (5, 2)
    assert out["w_raw"].shape == (5, 5)
    print("from HDVClassnNet import PPNet works")
    print("out keys:", sorted(out.keys()))


if __name__ == "__main__":
    main()
    
