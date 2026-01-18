import numpy as np
import pandas as pd
from typing import Tuple
###################### IO ######################
def load_points(csv: str, cols: Tuple[str,str,str]=("middle_x","middle_y","middle_z")) -> np.ndarray:
    """
    Load 3D points from CSV using provided column names (default to our current data's middle_x/y/z).
    """
    df = pd.read_csv(csv)
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {csv}. Available: {list(df.columns)[:12]}...")
    P = df[list(cols)].dropna().values.astype(np.float32)
    if len(P) < 50:
        raise ValueError(f"Too few points in {csv} after dropna on {cols}")
    return P