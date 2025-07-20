import numpy as np
import tensorflow as tf
from scipy.stats import qmc
from core.geometry.geometry2d import Geometry2D
from sklearn.model_selection import train_test_split

class ConvectionDiffusionDataset:
    """
    Generates dataset of collocation, initial, and boundary condition points for PINNs.
    Uses the domain geometry defined in Geometry2D.
    """

    def __init__(self):
        self.geometry = Geometry2D()
        self.bounds = self.geometry.domain_bounds()

        # Sampling settings
        self.N_r = 10000  # Collocation
        self.N_i = 400    # Initial condition
        self.N_b = 400    # Boundary condition

    def initial_condition(self, x, y):
        """Gaussian initial condition centered at (0.5, 0.5)"""
        return np.exp(-50 * ((x - 0.5)**2 + (y - 0.5)**2))
    
    def generate(self):
        x_min, x_max = self.bounds['x_min'], self.bounds['x_max']
        y_min, y_max = self.bounds['y_min'], self.bounds['y_max']
        t_min, t_max = self.bounds['t_min'], self.bounds['t_max']

        # Collocation points using LHS
        sampler = qmc.LatinHypercube(d=3)
        samples = sampler.random(self.N_r)
        x_r = samples[:, 0:1] * (x_max - x_min) + x_min
        y_r = samples[:, 1:2] * (y_max - y_min) + y_min
        t_r = samples[:, 2:3] * (t_max - t_min) + t_min

        res = np.hstack([x_r, y_r, t_r])  # Shape: (N_r, 3)

        # Initial condition points (t=0)
        x_i = np.random.uniform(x_min, x_max, (self.N_i, 1))
        y_i = np.random.uniform(y_min, y_max, (self.N_i, 1))
        t_i = np.zeros_like(x_i)
        u_i = self.initial_condition(x_i, y_i)

        in_i = np.hstack([x_i, y_i, t_i])  # Shape: (N_i, 3)
        out_i = u_i                        # Shape: (N_i, 1)

        # Boundary condition points
        x_b1 = np.ones((self.N_b // 4, 1)) * x_min
        x_b2 = np.ones((self.N_b // 4, 1)) * x_max
        x_b3 = np.random.uniform(x_min, x_max, (self.N_b // 4, 1))
        x_b4 = np.random.uniform(x_min, x_max, (self.N_b // 4, 1))
        y_b1 = np.random.uniform(y_min, y_max, (self.N_b // 4, 1))
        y_b2 = np.random.uniform(y_min, y_max, (self.N_b // 4, 1))
        y_b3 = np.ones((self.N_b // 4, 1)) * y_min
        y_b4 = np.ones((self.N_b // 4, 1)) * y_max
        t_b = np.random.uniform(t_min, t_max, (self.N_b // 4, 1))

        x_b = np.vstack([x_b1, x_b2, x_b3, x_b4])   # shape: (N_b, 1)
        y_b = np.vstack([y_b1, y_b2, y_b3, y_b4])   # shape: (N_b, 1)
        t_b = np.vstack([t_b, t_b, t_b, t_b])       # shape: (N_b, 1)
        u_b = np.zeros((self.N_b, 1))

        in_b = np.hstack([x_b, y_b, t_b])  # Shape: (N_b, 3)
        out_b = u_b                        # Shape: (N_b, 1)

        # -------- Convert to Tensors --------
        res = tf.convert_to_tensor(res, dtype=tf.float32)
        in_i = tf.convert_to_tensor(in_i, dtype=tf.float32)
        out_i = tf.convert_to_tensor(out_i, dtype=tf.float32)
        in_b = tf.convert_to_tensor(in_b, dtype=tf.float32)
        out_b = tf.convert_to_tensor(out_b, dtype=tf.float32)

        # Convert tensors to NumPy if they're TensorFlow tensors
        in_i_np = in_i.numpy() if tf.is_tensor(in_i) else in_i
        out_i_np = out_i.numpy() if tf.is_tensor(out_i) else out_i
        in_b_np = in_b.numpy() if tf.is_tensor(in_i) else in_b
        out_b_np = out_b.numpy() if tf.is_tensor(out_i) else out_b

        in_i_train, in_i_test, out_i_train, out_i_test = train_test_split(in_i_np, out_i_np, test_size=0.1, random_state=42)
        in_b_train, in_b_test, out_b_train, out_b_test = train_test_split(in_b_np, out_b_np, test_size=0.1, random_state=42)

        return (
            res,
            in_i_train, in_i_test, out_i_train, out_i_test,
            in_b_train, in_b_test, out_b_train, out_b_test
        )