"""
Modified FNO implementation that decomposes the solution operator using its principal symbol
"""
from scipy import integrate
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

        # Apply operator in Fourier space
        f_fourier = torch.fft.fft2(f)
        Qf_fourier = f_fourier * symbol

        # The factor 1/N² is for the inverse FFT normalization
        Qf = torch.fft.ifft2(Qf_fourier).real / (self.grid_size ** 2)

        return Qf

    # def apply_principal_symbol_operator(self, f):
    #     """
    #     Applies the pseudodifferential operator Q with symbol 1/|ξ|² to input f
    #     Uses FFT to implement the operation in Fourier space
    #     """
    #     batch_size = f.shape[0]
    #
    #     # Get frequencies
    #     freq_x = torch.fft.fftfreq(self.grid_size, 1 / self.grid_size).to(
    #         self.device)
    #     freq_y = torch.fft.fftfreq(self.grid_size, 1 / self.grid_size).to(
    #         self.device)
    #     freq_x, freq_y = torch.meshgrid(freq_x, freq_y, indexing='ij')
    #
    #     # Compute |ξ|²
    #     xi_squared = freq_x ** 2 + freq_y ** 2
    #
    #     # Add small epsilon to avoid division by zero
    #     symbol = 1.0 / (xi_squared + 1e-10)
    #
    #     # Zero out the zero frequency component to match zero mean
    #     symbol[0, 0] = 0
    #
    #     # Reshape for broadcasting
    #     symbol = symbol.reshape(1, 1, self.grid_size, self.grid_size)
    #
    #     # Apply operator in Fourier space
    #     f_fourier = torch.fft.fft2(f)
    #     Qf_fourier = f_fourier * symbol
    #
    #     # Transform back to physical space
    #     Qf = torch.fft.ifft2(Qf_fourier).real
    #
    #     return Qf

    # def get_Qf(f_func, z, grid_size=100, eps=1e-10):
    #     """
    #     Compute Q(f)(z) = -(1/2π) ∫ f(y)log|z-y|dy using numerical integration
    #
    #     Args:
    #         f_func: Function that takes (x,y) coordinates and returns value
    #         z: Point (z1,z2) at which to evaluate Q(f)
    #         grid_size: Number of points for quadrature
    #         eps: Small parameter for handling log singularity
    #
    #     Returns:
    #         Value of Q(f) at point z
    #     """
    #
    #     def integrand(x, y):
    #         diff_x = x - z[0]
    #         diff_y = y - z[1]
    #         r_squared = diff_x ** 2 + diff_y ** 2
    #
    #         # Handle log singularity by adding small eps
    #         log_term = np.log(np.sqrt(r_squared + eps))
    #
    #         return -1 / (2 * np.pi) * f_func(x, y) * log_term
    #
    #     # Define integration regions to handle singularity
    #     # Split into regions away from and near the singularity
    #     def get_regions():
    #         z1, z2 = z
    #         delta = 0.1  # Size of region around singularity
    #
    #         regions = []
    #
    #         # Add regions avoiding the singularity point
    #         if z1 - delta > 0:
    #             regions.append([[0, z1 - delta], [0, 1]])
    #         if z1 + delta < 1:
    #             regions.append([[z1 + delta, 1], [0, 1]])
    #         if z2 - delta > 0:
    #             regions.append(
    #                 [[max(0, z1 - delta), min(1, z1 + delta)], [0, z2 - delta]])
    #         if z2 + delta < 1:
    #             regions.append(
    #                 [[max(0, z1 - delta), min(1, z1 + delta)], [z2 + delta, 1]])
    #
    #         return regions
    #
    #     # Integrate over regions away from singularity
    #     result = 0
    #     for region in get_regions():
    #         [[x1, x2], [y1, y2]] = region
    #         integral, _ = integrate.dblquad(
    #             integrand,
    #             y1, y2,  # y limits
    #             lambda y: x1, lambda y: x2,  # x limits
    #             epsabs=1e-6, epsrel=1e-6
    #         )
    #         result += integral
    #
    #     # Handle singular region using polar coordinates
    #     def polar_integrand(r, theta):
    #         x = z[0] + r * np.cos(theta)
    #         y = z[1] + r * np.sin(theta)
    #
    #         # Only include points in [0,1]²
    #         if not (0 <= x <= 1 and 0 <= y <= 1):
    #             return 0
    #
    #         return -1 / (2 * np.pi) * r * f_func(x, y) * np.log(r)
    #
    #     # Integrate over small circle around singularity
    #     delta = 0.1
    #     r_integral, _ = integrate.dblquad(
    #         polar_integrand,
    #         0, 2 * np.pi,  # theta limits
    #         lambda theta: eps, lambda theta: delta,  # r limits
    #         epsabs=1e-6, epsrel=1e-6
    #     )
    #
    #     result += r_integral
    #
    #     return result
    #
    # def compute_Qf_grid(f_func, grid_size=32):
    #     """
    #     Compute Q(f) on a grid of points
    #
    #     Args:
    #         f_func: Function that takes (x,y) coordinates and returns value
    #         grid_size: Number of grid points in each dimension
    #
    #     Returns:
    #         2D array of Q(f) values on grid
    #     """
    #     x = np.linspace(0, 1, grid_size)
    #     y = np.linspace(0, 1, grid_size)
    #     Qf = np.zeros((grid_size, grid_size))
    #
    #     for i in range(grid_size):
    #         for j in range(grid_size):
    #             z = [x[i], y[j]]
    #             Qf[i, j] = get_Qf(f_func, z, grid_size)
    #             if (i * grid_size + j) % 100 == 0:
    #                 print(
    #                     f"Computing Q(f): {(i * grid_size + j) / (grid_size ** 2) * 100:.1f}% complete")
    #
    #     return Qf

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

        # Apply principal symbol operator and compute residuals
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
        """
        Analyze the residuals after principal part extraction
        """
        # Ensure correct dimensions
        if len(x_data.shape) == 3:
            x_data = x_data.unsqueeze(1)
        if len(y_data.shape) == 3:
            y_data = y_data.unsqueeze(1)

        # Move data to device
        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)

        # Calculate Q(f)
        Qf = self.apply_principal_symbol_operator(x_data)

        # Calculate residuals
        residuals = y_data - Qf

        # Get random samples
        indices = np.random.choice(len(x_data), num_samples, replace=False)

        for i, idx in enumerate(indices):
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Reshape data for plotting
            y_plot = y_data[idx, 0].cpu().reshape(self.grid_size,
                                                  self.grid_size)
            Qf_plot = Qf[idx, 0].cpu().reshape(self.grid_size, self.grid_size)
            residual_plot = residuals[idx, 0].cpu().reshape(self.grid_size,
                                                            self.grid_size)

            # Plot original solution
            im1 = axes[0, 0].imshow(y_plot, cmap='viridis')
            axes[0, 0].set_title(f'Original Solution u')
            plt.colorbar(im1, ax=axes[0, 0])

            # Plot principal part Q(f)
            im2 = axes[0, 1].imshow(Qf_plot, cmap='viridis')
            axes[0, 1].set_title(f'Principal Part Q(f)')
            plt.colorbar(im2, ax=axes[0, 1])

            # Plot residual
            im3 = axes[1, 0].imshow(residual_plot, cmap='viridis')
            axes[1, 0].set_title(f'Residual (u - Q(f))')
            plt.colorbar(im3, ax=axes[1, 0])

            # Plot frequency content of residual
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
            print(
                f"Residual relative size: {residual_norm / solution_norm:.6f}")

            # Print max/min values
            print(f"Solution range: [{y_plot.min():.6f}, {y_plot.max():.6f}]")
            print(
                f"Principal part range: [{Qf_plot.min():.6f}, {Qf_plot.max():.6f}]")
            print(
                f"Residual range: [{residual_plot.min():.6f}, {residual_plot.max():.6f}]")

            # Analyze frequency bands of residual
            freq_x = torch.fft.fftfreq(self.grid_size)
            freq_y = torch.fft.fftfreq(self.grid_size)
            freq_x, freq_y = torch.meshgrid(freq_x, freq_y, indexing='ij')
            freqs = torch.sqrt(freq_x ** 2 + freq_y ** 2)

            # Define frequency bands
            bands = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

            print("\nFrequency band analysis of residual:")
            for low, high in bands:
                mask = (freqs >= low) & (freqs < high)
                band_energy = torch.sum(torch.abs(fft_residual[mask]) ** 2)
                print(
                    f"Band [{low:.1f}, {high:.1f}]: Energy = {band_energy:.6f}")

    def train_fno(self, x_train, residual_train, Qf_train,
                  x_test, residual_test, Qf_test, n_epochs=50):
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

            # Plot input function f
            im1 = ax1.imshow(x_test[idx, 0].cpu(), cmap='viridis')
            ax1.set_title(f'Input f_{i + 1}')
            plt.colorbar(im1, ax=ax1)

            # Plot principal part Q(f)
            im2 = ax2.imshow(Qf_test[idx, 0].cpu(), cmap='viridis')
            ax2.set_title(f'Principal Part Q(f)_{i + 1}')
            plt.colorbar(im2, ax=ax2)

            # Plot true solution u
            im3 = ax3.imshow(y_test[idx, 0].cpu(), cmap='viridis')
            ax3.set_title(f'True Solution u_{i + 1}')
            plt.colorbar(im3, ax=ax3)

            # Plot predicted solution
            im4 = ax4.imshow(pred_full[idx, 0].cpu(), cmap='viridis')
            ax4.set_title(f'Predicted Solution û_{i + 1}')
            plt.colorbar(im4, ax=ax4)

            plt.tight_layout()
            plt.show()

            # Compute relative errors
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
        # Remove channel dimension
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        Qf = Qf.squeeze()

        # Compute 2D FFT
        fft_true = torch.fft.fft2(y_true.cpu())
        fft_pred = torch.fft.fft2(y_pred.cpu())
        fft_Qf = torch.fft.fft2(Qf.cpu())

        # Compute magnitude spectra
        mag_true = torch.abs(fft_true)
        mag_pred = torch.abs(fft_pred)
        mag_Qf = torch.abs(fft_Qf)

        # Plot frequency content
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

##############TEST##############
    def test_principal_symbol(self):
        """
        Test the principal symbol operator on a known example
        """
        # Create a test function f(x,y) = sin(2πx)sin(2πy)
        x = torch.linspace(0, 1, self.grid_size)
        y = torch.linspace(0, 1, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        f = torch.sin(2 * np.pi * X) * torch.sin(2 * np.pi * Y)
        f = f.unsqueeze(0).unsqueeze(0).to(self.device)

        # Apply Q operator
        Qf = self.apply_principal_symbol_operator(f)

        # True solution for -Δu = f is u = f/(8π²)
        u_true = f / (4 * np.pi ** 2)

        # Compare
        error = torch.norm(Qf - u_true) / torch.norm(u_true)
        print(f"Test error: {error:.6f}")
        print(f"Qf norm: {torch.norm(Qf):.6f}")
        print(f"True solution norm: {torch.norm(u_true):.6f}")

###############################


def main():
    # Initialize tester
    tester = FNOTesterWithPrincipalSymbol(data_points=150, grid_size=64,
                                          truncation_order=10)

    # Generate/load data
    print("Generating/loading data...")
    x_data, y_data = tester.generate_data()

    # Analyze residuals before training
    print("\nAnalyzing residuals...")
    tester.analyze_residuals(x_data, y_data)

    # Prepare data with principal symbol decomposition
    print("Preparing data...")
    (x_train, y_train, Qf_train, residual_train,
     x_test, y_test, Qf_test, residual_test) = tester.prepare_data(x_data,
                                                                   y_data)

    # Train model on residual
    print("Training FNO...")
    model, train_losses, test_losses = tester.train_fno(
        x_train, residual_train, Qf_train,
        x_test, residual_test, Qf_test
    )

    # Plot results
    print("Plotting comparisons...")
    tester.plot_comparison(model, x_test, y_test, Qf_test)

    # Analyze frequencies
    print("Analyzing frequency content...")
    with torch.no_grad():
        pred_residual = model(x_test[:1])
        pred_full = pred_residual + Qf_test[:1]
    tester.analyze_frequencies(y_test[0], pred_full[0], Qf_test[0])

##############TEST##############
    # # Test principal symbol operator
    # print("Testing principal symbol operator...")
    # tester.test_principal_symbol()

###############################

if __name__ == "__main__":
    main()