from itertools import product
from os import makedirs
from pathlib import Path

from numpy import arange
from yaml import dump

if __name__ == "__main__":
    target_dir = Path("cfgs/run2")
    makedirs(target_dir, exist_ok=True)
    
    d_multiplier = arange(4, 10, 0.5)
    n_blocks = arange(3, 5, 1)
    
    for idx, comb in enumerate(product(d_multiplier, n_blocks)):
        cfg = {
            "model_name": "conv1d",
            "recurrent": False,
            "framed": False,
            "network_config": {
                "stride_res": int(320),
                "depth_multiplier": float(comb[0]),
                "num_blocks": int(comb[1]),
                "downsampling_layers": list([4])
            }
        }
        
        with open(target_dir.joinpath(f"dnet_{idx}.yaml"), "w") as write:
            dump(cfg, write)
