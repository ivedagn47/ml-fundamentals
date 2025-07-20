import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.model.model import PINN
from core.physics.physics import compute_loss
from core.data.dataset import ConvectionDiffusionDataset


class PINNsTrainer:
    def __init__(self, epochs=100000, batch_size=10, output_dir="output/exp2"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = PINN()

        (
            self.res,
            self.in_i_train, self.in_i_test,
            self.out_i_train, self.out_i_test,
            self.in_b_train, self.in_b_test,
            self.out_b_train, self.out_b_test
        )= ConvectionDiffusionDataset().generate()

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=500,
            decay_rate=0.95,
            staircase=True
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    def train_step(self):
        with tf.GradientTape() as tape:
            total_loss, loss_r, loss_b, loss_i = compute_loss(
                self.model, self.res,
                self.in_i_train, self.out_i_train,
                self.in_b_train, self.out_b_train
            )
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        clipped_grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
        self.optimizer.apply_gradients(zip(clipped_grads, self.model.trainable_variables))
        return total_loss, loss_r, loss_b, loss_i

    def train(self):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            total_loss, loss_r, loss_b, loss_i = self.train_step()
            train_losses.append(total_loss.numpy())

            val_pred = self.model(self.in_i_test, training=False)
            val_loss = tf.reduce_mean(tf.square(val_pred - self.out_b_test)).numpy()
            val_losses.append(val_loss)

            print(f"Epoch {epoch:02d} | Train Loss: {total_loss:.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                #self.model.save_weights(os.path.join(self.output_dir, "best_model_weights.h5"))


        self.save_results(train_losses, val_losses)

    def save_results(self, train_losses, val_losses):
        u_pred_val = self.model(self.in_i_test, training=False).numpy()
        u_true_val = np.array(self.out_i_test)

        u_true_val = u_true_val.reshape(5, 8)
        u_pred_val = u_pred_val.reshape(5, 8)

        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train MSE")
        plt.plot(val_losses, label="Validation MSE")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
        print(f"Saved loss curve at: {os.path.join(self.output_dir, 'loss_curve.png')}")

        error = np.abs(u_pred_val - u_true_val)
        l2_error = np.sqrt(np.mean((u_pred_val - u_true_val) ** 2))
        l1_error = np.mean(np.abs(u_pred_val - u_true_val))
        l_inf_error = np.max(np.abs(u_pred_val - u_true_val))
        rel_l2_error = l2_error / np.sqrt(np.mean(u_true_val ** 2))
        rel_l1_error = l1_error / np.mean(np.abs(u_true_val))
        rel_l_inf_error = l_inf_error / np.max(np.abs(u_true_val))

        error_df = pd.DataFrame(
    {
        "L2 Error": [l2_error],
        "L1 Error": [l1_error],
        "L_inf Error": [l_inf_error],
        "Relative L2 Error": [rel_l2_error],
        "Relative L1 Error": [rel_l1_error],
        "Relative L_inf Error": [rel_l_inf_error],
    }
)

        error_df.to_csv(f"{self.output_dir}/error_metrics.csv", index=False)
        print(error_df)

        # Check for square reshape
        '''total_points = u_true_val.shape[0]
        N = int(np.sqrt(total_points))
        assert N * N == total_points, f"Cannot reshape {total_points} points to square grid"

        u_true_grid = u_true_val.reshape(N, N)
        u_pred_grid = u_pred_val.reshape(N, N)
        error_grid = error.reshape(N, N)'''

        plt.figure(figsize=(7, 5))
        plt.subplot(121)
        plt.imshow(u_true_val, cmap="viridis")
        plt.title("True u(x, y)")
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(u_pred_val, cmap="viridis")
        plt.title("Predicted u(x, y)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pred_vs_true.png"))

        plt.figure(figsize=(5, 5))
        plt.title("Point-wise Absolute Error")
        plt.imshow(error, cmap='Reds')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "error_plot.png"))

        print("All plots saved in 'output'")


if __name__ == "__main__":
    trainer = PINNsTrainer()
    trainer.train()