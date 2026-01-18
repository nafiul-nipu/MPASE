import numpy as np
import mpase

rng = np.random.default_rng(0)

A = rng.normal(size=(500, 3)).astype(np.float32)
R = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float32)
B = (A @ R.T) + 0.05 * rng.normal(size=A.shape).astype(np.float32)

result = mpase.mpase(
    points_list=[A, B],
    labels=["A", "B"],
    align_mode="auto",
    run_hdr=True,
    run_pf=True,
    out_dir="mpase_output"
)

print(result["metrics"].head())
mpase.view(result, kind="hdr", plane="XY", levels=[95, 80])
