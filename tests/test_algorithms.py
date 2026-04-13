import numpy as np
from algorithms.dirichlet import fixed_point_dirichlet


def test_fixed_point_small():
    # small synthetic counts with clear proportions
    counts = np.array([[10, 5, 0, 0], [9,6,1,0], [11,4,0,1]])
    alpha, info = fixed_point_dirichlet(counts, tol=1e-6, max_iter=200)
    assert alpha.shape[0] == 4
    assert info['converged'] in (True, False)
    # alpha should be positive
    assert (alpha > 0).all()

