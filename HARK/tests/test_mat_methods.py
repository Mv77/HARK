import unittest

import numpy as np

from HARK.mat_methods import mass_to_grid, transition_mat
from HARK.utilities import jump_to_grid_1D, jump_to_grid_2D


# Compare general mass_to_grid with jump_to_grid_1D
class TestMassToGrid1D(unittest.TestCase):
    def setUp(self):
        n_grid = 30
        n_points = 1000

        # Create 1d grid
        self.grid = np.linspace(0, 10.0, n_grid)
        # Simulate 1d points from a normal distribution
        # large variance to ensure some points are outside the grid
        self.points = np.random.normal(5, 20, n_points)
        # Create weights
        self.weights = np.random.uniform(0, 1, n_points)

    def test_compare_jump_to_grid_1d(self):
        # Compare mass_to_grid with jump_to_grid_1D
        res1 = mass_to_grid(self.points[..., np.newaxis], self.weights, (self.grid,))
        res2 = jump_to_grid_1D(self.points, self.weights, self.grid)

        # Compare results
        self.assertTrue(np.allclose(res1, res2))


# Compare general mass_to_grid with jump_to_grid_2D
class TestMassToGrid2D(unittest.TestCase):
    def setUp(self):
        n_grid = 30
        n_points = 1000

        # Create 2d grid
        self.x_grid = np.linspace(0.0, 10, n_grid)
        self.y_grid = np.linspace(0.0, 10, n_grid)

        # Simulate 2d points from a normal distribution
        # large variance to ensure some points are outside the grid
        mean = np.array([3, 4])
        vcov = np.array([[10.0, -0.5], [-0.5, 3.0]])
        self.points = np.random.multivariate_normal(mean, vcov, n_points)
        self.weights = np.repeat(1.3, n_points)

    def test_compare_jump_to_grid_2d(self):
        # Compare mass_to_grid with jump_to_grid_2D
        res1 = mass_to_grid(self.points, self.weights, (self.x_grid, self.y_grid))
        res2 = jump_to_grid_2D(
            self.points[:, 0], self.points[:, 1], self.weights, self.x_grid, self.y_grid
        )

        # Compare results
        self.assertTrue(np.allclose(res1, res2))


class Test3DMassToGrid(unittest.TestCase):
    def test_simple_3d(self):
        # 3d grid of 2 points in each dimension
        x_grid = np.array([0.0, 1.0])
        y_grid = np.array([0.0, 1.0])
        z_grid = np.array([0.0, 1.0])

        # Some points
        my_points = np.array(
            [
                [0.5, 0.5, 0.5],
                [3.0, 3.0, 3.0],
            ]
        )
        mass = np.array([1.0, 1.0])

        # Compare results
        grid_mass = mass_to_grid(my_points, mass, (x_grid, y_grid, z_grid))

        grid_mass = grid_mass.reshape((2, 2, 2))

        # Check the mass on the 8 points
        self.assertTrue(grid_mass[0, 0, 0] == 1 / 8)
        self.assertTrue(grid_mass[0, 0, 1] == 1 / 8)
        self.assertTrue(grid_mass[0, 1, 0] == 1 / 8)
        self.assertTrue(grid_mass[0, 1, 1] == 1 / 8)
        self.assertTrue(grid_mass[1, 0, 0] == 1 / 8)
        self.assertTrue(grid_mass[1, 0, 1] == 1 / 8)
        self.assertTrue(grid_mass[1, 1, 0] == 1 / 8)
        self.assertTrue(grid_mass[1, 1, 1] == (1 / 8 + 1.0))


class TestTransMatMultiplication(unittest.TestCase):
    def setUp(self):
        # Create dummy 2-gridpoint problems

        # A newborn "distribution"
        self.newborn_dstn = np.array([0.5, 0.3, 0.2])

        # Infinite horizon transition
        inf_h_trans = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]])
        self.inf_horizon_mat = transition_mat(
            living_tmat=inf_h_trans,
            surv_prob=0.9,
            newborn_dstn=self.newborn_dstn,
        )

    def test_post_multiply(self):
        nrows = self.inf_horizon_mat.grid_len
        mat = np.random.rand(nrows, 3)
        res = self.inf_horizon_mat.post_multiply(mat)
        res2 = np.dot(self.inf_horizon_mat.get_full_tmat(), mat)
        self.assertTrue(np.allclose(res, res2))

    def test_pre_multiply(self):
        ncols = self.inf_horizon_mat.grid_len
        mat = np.random.rand(3, ncols)
        res = self.inf_horizon_mat.pre_multiply(mat)
        res2 = np.dot(mat, self.inf_horizon_mat.get_full_tmat())
        self.assertTrue(np.allclose(res, res2))

    def test_steady_state(self):
        # Compute using iterative class method
        ss_iter = self.inf_horizon_mat.find_steady_state_dstn()
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(self.inf_horizon_mat.get_full_tmat().T)
        # Find the eigenvector corresponding to the eigenvalue 1
        ss_eig = eigvecs[:, np.isclose(eigvals, 1.0)]
        # Normalize the eigenvector
        ss_eig = ss_eig / ss_eig.sum()
        # Compare the two
        self.assertTrue(np.allclose(ss_iter, ss_eig, atol=1e-8))
