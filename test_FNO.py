"""
The neuraloperator library's doc:
https://neuraloperator.github.io/dev/index.html
"""

from neuralop.models import FNO2d
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from synthetic_data_for_neural_operators.code.main import DataGenerator
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


class FNOTester:
    def __init__(self, data_points, grid_size, truncation_order):
        self.data_points = data_points
        self.grid_size = grid_size
        self.truncation_order = truncation_order
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_data(self):
        """
        Generate data using paper "generating synthetic data for operator learning" method
        """
        x_path = f'synthetic_data_dirichlet_{self.data_points}_x.pt'
        y_path = f'synthetic_data_dirichlet_{self.data_points}_y.pt'

        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            generator = DataGenerator(
                data_points=self.data_points,
                grid_size=self.grid_size,
                truncation_order=self.truncation_order,
                x_save_path=x_path,
                y_save_path=y_path
            )
            generator.generate()

        x_data = torch.load(x_path).type(torch.FloatTensor)
        y_data = torch.load(y_path).type(torch.FloatTensor)

        print("Data shapes after loading:")
        print("x_data shape:", x_data.shape)
        print("y_data shape:", y_data.shape)

        return x_data, y_data

    def prepare_data(self, x_data, y_data):

        n_train = int(0.9 * self.data_points)

        if len(y_data.shape) == 3:
            y_data = y_data.unsqueeze(1)

        if len(x_data.shape) == 3:
            x_data = x_data.unsqueeze(1)

        x_train = x_data[:n_train].to(self.device)
        y_train = y_data[:n_train].to(self.device)
        x_test = x_data[n_train:].to(self.device)
        y_test = y_data[n_train:].to(self.device)

        print("\nTraining data shapes:")
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)

        return x_train, y_train, x_test, y_test

    def train_fno(self, x_train, y_train, x_test, y_test, n_epochs=150):
        model = FNO2d(
            n_modes_height=16,
            n_modes_width=16,
            hidden_channels=64,
            in_channels=1,
            out_channels=1
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, n_epochs)

        train_losses = []
        test_losses = []

        for epoch in range(n_epochs):

            if epoch % 5 == 0:
                print(f"Training epoch {epoch}/{n_epochs}")

            model.train()
            optimizer.zero_grad()

            pred = model(x_train)
            loss = F.mse_loss(pred, y_train)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_pred = model(x_test)
                    test_loss = F.mse_loss(test_pred, y_test).item()
                    train_loss = loss.item()

                    train_losses.append(train_loss)
                    test_losses.append(test_loss)

                    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

        return model, train_losses, test_losses

    def plot_comparison(self, model, x_test, y_test, num_samples=5):
        model.eval()
        with torch.no_grad():
            predictions = model(x_test)

        indices = np.random.choice(len(x_test), num_samples, replace=False)

        for i, idx in enumerate(indices):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

            im1 = ax1.imshow(x_test[idx, 0].cpu(), cmap='viridis')
            ax1.set_title(f'Input f_{i+1}')
            plt.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(y_test[idx, 0].cpu(), cmap='viridis')
            ax2.set_title(f'True Solution u_{i+1}')
            plt.colorbar(im2, ax=ax2)

            im3 = ax3.imshow(predictions[idx, 0].cpu(), cmap='viridis')
            ax3.set_title(f'Predicted Solution û_{i+1}')
            plt.colorbar(im3, ax=ax3)

            plt.tight_layout()
            plt.show()

            rel_error = torch.norm(predictions[idx] - y_test[idx]) / torch.norm(y_test[idx])
            print(f'Sample {i+1} Relative L2 Error: {rel_error:.6f}')

    def analyze_frequencies(self, y_true, y_pred):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()

        fft_true = torch.fft.fft2(y_true.cpu())
        fft_pred = torch.fft.fft2(y_pred.cpu())

        mag_true = torch.abs(fft_true)
        mag_pred = torch.abs(fft_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        im1 = ax1.imshow(torch.log(mag_true + 1e-10))
        ax1.set_title('True Solution Frequency Content')
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(torch.log(mag_pred + 1e-10))
        ax2.set_title('Predicted Solution Frequency Content')
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.show()


def main():
    tester = FNOTester(data_points=1e4, grid_size=64, truncation_order=10)

    print("Generating/loading data...")
    x_data, y_data = tester.generate_data()

    print("Preparing data...")
    x_train, y_train, x_test, y_test = tester.prepare_data(x_data, y_data)

    print("Training FNO...")
    model, train_losses, test_losses = tester.train_fno(x_train, y_train,
                                                        x_test, y_test)
    print("Plotting comparisons...")
    tester.plot_comparison(model, x_test, y_test)

    print("Analyzing frequency content...")
    with torch.no_grad():
        pred = model(x_test[:1])
    tester.analyze_frequencies(y_test[0], pred[0])


if __name__ == "__main__":
    main()