"""
Modified FNO implementation that decomposes the solution operator using its principal symbol
"""
from neuralop.models import FNO2d
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from synthetic_data_for_neural_operators.code.main import DataGenerator
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


class FNOTesterWithPrincipalSymbol:
    def __init__(self, data_points, grid_size, truncation_order):
        self.data_points = data_points
        self.grid_size = grid_size
        self.truncation_order = truncation_order
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def apply_principal_symbol_operator(self, f):
        """
        For the example we study here:
        Apply the pseudodifferential operator Q with symbol 1/|ξ|² to input f
        Use FFT to implement the operation in Fourier space (with proper scaling)
        """

        freq_x = 2 * np.pi * torch.fft.fftfreq(self.grid_size).to(self.device)
        freq_y = 2 * np.pi * torch.fft.fftfreq(self.grid_size).to(self.device)
        freq_x, freq_y = torch.meshgrid(freq_x, freq_y, indexing='ij')

        xi_squared = freq_x ** 2 + freq_y ** 2

        symbol = 1.0 / ((xi_squared + 1e-6))

        symbol[0, 0] = 0

        symbol = symbol.reshape(1, 1, self.grid_size, self.grid_size)

        f_fourier = torch.fft.fft2(f)
        Qf_fourier = f_fourier * symbol

        Qf = torch.fft.ifft2(Qf_fourier).real / (self.grid_size ** 2)

        return Qf


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

        return x_data, y_data

    def prepare_data(self, x_data, y_data):

        n_train = int(0.9 * self.data_points)

        if len(y_data.shape) == 3:
            y_data = y_data.unsqueeze(1)
        if len(x_data.shape) == 3:
            x_data = x_data.unsqueeze(1)

        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)

        Qf_data = self.apply_principal_symbol_operator(x_data)
        residual_data = y_data - Qf_data

        x_train = x_data[:n_train]
        y_train = y_data[:n_train]
        Qf_train = Qf_data[:n_train]
        residual_train = residual_data[:n_train]

        x_test = x_data[n_train:]
        y_test = y_data[n_train:]
        Qf_test = Qf_data[n_train:]
        residual_test = residual_data[n_train:]

        return (x_train, y_train, Qf_train, residual_train,
                x_test, y_test, Qf_test, residual_test)

    def analyze_residuals(self, x_data, y_data, num_samples=5):

        if len(x_data.shape) == 3:
            x_data = x_data.unsqueeze(1)
        if len(y_data.shape) == 3:
            y_data = y_data.unsqueeze(1)

        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)

        Qf = self.apply_principal_symbol_operator(x_data)

        residuals = y_data - Qf

        indices = np.random.choice(len(x_data), num_samples, replace=False)

        for i, idx in enumerate(indices):
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            y_plot = y_data[idx, 0].cpu().reshape(self.grid_size,
                                                  self.grid_size)
            Qf_plot = Qf[idx, 0].cpu().reshape(self.grid_size, self.grid_size)
            residual_plot = residuals[idx, 0].cpu().reshape(self.grid_size,
                                                            self.grid_size)

            im1 = axes[0, 0].imshow(y_plot, cmap='viridis')
            axes[0, 0].set_title(f'Original Solution u')
            plt.colorbar(im1, ax=axes[0, 0])

            im2 = axes[0, 1].imshow(Qf_plot, cmap='viridis')
            axes[0, 1].set_title(f'Principal Part Q(f)')
            plt.colorbar(im2, ax=axes[0, 1])

            im3 = axes[1, 0].imshow(residual_plot, cmap='viridis')
            axes[1, 0].set_title(f'Residual (u - Q(f))')
            plt.colorbar(im3, ax=axes[1, 0])

            fft_residual = torch.fft.fft2(residual_plot)
            freq_plot = torch.log(torch.abs(fft_residual) + 1e-10)
            im4 = axes[1, 1].imshow(freq_plot, cmap='viridis')
            axes[1, 1].set_title('Residual Frequency Content')
            plt.colorbar(im4, ax=axes[1, 1])

            plt.tight_layout()
            plt.show()

            solution_norm = torch.norm(y_data[idx])
            principal_norm = torch.norm(Qf[idx])
            residual_norm = torch.norm(residuals[idx])

            print(f"\nSample {i + 1} (index {idx}):")
            print(f"Solution norm: {solution_norm:.6f}")
            print(f"Principal part norm: {principal_norm:.6f}")
            print(f"Residual norm: {residual_norm:.6f}")

    def train_fno(self, x_train, residual_train, Qf_train,
                  x_test, residual_test, Qf_test, n_epochs=150):
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

            pred_residual = model(x_train)

            pred_full = pred_residual + Qf_train
            loss = F.mse_loss(pred_full, residual_train + Qf_train)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_pred_residual = model(x_test)
                    test_pred_full = test_pred_residual + Qf_test
                    test_loss = F.mse_loss(test_pred_full,
                                           residual_test + Qf_test).item()
                    train_loss = loss.item()

                    train_losses.append(train_loss)
                    test_losses.append(test_loss)

                    print(
                        f'Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

        return model, train_losses, test_losses

    def plot_comparison(self, model, x_test, y_test, Qf_test, num_samples=5):
        model.eval()
        with torch.no_grad():
            pred_residual = model(x_test)
            pred_full = pred_residual + Qf_test

        indices = np.random.choice(len(x_test), num_samples, replace=False)

        for i, idx in enumerate(indices):
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4))

            im1 = ax1.imshow(x_test[idx, 0].cpu(), cmap='viridis')
            ax1.set_title(f'Input f_{i + 1}')
            plt.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(Qf_test[idx, 0].cpu(), cmap='viridis')
            ax2.set_title(f'Principal Part Q(f)_{i + 1}')
            plt.colorbar(im2, ax=ax2)

            im3 = ax3.imshow(y_test[idx, 0].cpu(), cmap='viridis')
            ax3.set_title(f'True Solution u_{i + 1}')
            plt.colorbar(im3, ax=ax3)

            im4 = ax4.imshow(pred_full[idx, 0].cpu(), cmap='viridis')
            ax4.set_title(f'Predicted Solution û_{i + 1}')
            plt.colorbar(im4, ax=ax4)

            plt.tight_layout()
            plt.show()

            full_error = torch.norm(pred_full[idx] - y_test[idx]) / torch.norm(
                y_test[idx])
            principal_error = torch.norm(
                Qf_test[idx] - y_test[idx]) / torch.norm(y_test[idx])
            residual_error = torch.norm(pred_residual[idx]) / torch.norm(
                y_test[idx])

            print(f'Sample {i + 1}:')
            print(f'  Full Solution Relative L2 Error: {full_error:.6f}')
            print(f'  Principal Part Relative L2 Error: {principal_error:.6f}')
            print(f'  Residual Relative L2 Error: {residual_error:.6f}')

    def analyze_frequencies(self, y_true, y_pred, Qf):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        Qf = Qf.squeeze()

        fft_true = torch.fft.fft2(y_true.cpu())
        fft_pred = torch.fft.fft2(y_pred.cpu())
        fft_Qf = torch.fft.fft2(Qf.cpu())

        mag_true = torch.abs(fft_true)
        mag_pred = torch.abs(fft_pred)
        mag_Qf = torch.abs(fft_Qf)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

        im1 = ax1.imshow(torch.log(mag_true + 1e-10))
        ax1.set_title('True Solution Frequency Content')
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(torch.log(mag_pred + 1e-10))
        ax2.set_title('Predicted Solution Frequency Content')
        plt.colorbar(im2, ax=ax2)

        im3 = ax3.imshow(torch.log(mag_Qf + 1e-10))
        ax3.set_title('Principal Part Frequency Content')
        plt.colorbar(im3, ax=ax3)

        plt.tight_layout()
        plt.show()


def main():
    tester = FNOTesterWithPrincipalSymbol(data_points=1e4, grid_size=64,
                                          truncation_order=10)

    print("Generating/loading data...")
    x_data, y_data = tester.generate_data()

    print("\nAnalyzing residuals...")
    tester.analyze_residuals(x_data, y_data)

    print("Preparing data...")
    (x_train, y_train, Qf_train, residual_train,
     x_test, y_test, Qf_test, residual_test) = tester.prepare_data(x_data,
                                                                   y_data)

    print("Training FNO...")
    model, train_losses, test_losses = tester.train_fno(
        x_train, residual_train, Qf_train,
        x_test, residual_test, Qf_test
    )

    print("Plotting comparisons...")
    tester.plot_comparison(model, x_test, y_test, Qf_test)

    print("Analyzing frequency content...")
    with torch.no_grad():
        pred_residual = model(x_test[:1])
        pred_full = pred_residual + Qf_test[:1]
    tester.analyze_frequencies(y_test[0], pred_full[0], Qf_test[0])

if __name__ == "__main__":
    main()