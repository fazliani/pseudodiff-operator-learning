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
from scipy.interpolate import RegularGridInterpolator


class FNOTesterWithPrincipalSymbol:
    def __init__(self, data_points=100, grid_size=85, truncation_order=20):
        self.data_points = data_points
        self.grid_size = grid_size
        self.truncation_order = truncation_order
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def get_Qf(self, f_func, z, eps=1e-10):
        """
        Compute Q(f)(z) = -(1/2π) ∫ f(y)log|z-y|dy using numerical integration
        """

        def integrand(x, y):
            diff_x = x - z[0]
            diff_y = y - z[1]
            r_squared = diff_x ** 2 + diff_y ** 2
            log_term = np.log(np.sqrt(r_squared + eps))
            return -1 / (2 * np.pi) * f_func(x, y) * log_term

        def get_regions():
            z1, z2 = z
            delta = 0.1
            regions = []
            if z1 - delta > 0:
                regions.append([[0, z1 - delta], [0, 1]])
            if z1 + delta < 1:
                regions.append([[z1 + delta, 1], [0, 1]])
            if z2 - delta > 0:
                regions.append(
                    [[max(0, z1 - delta), min(1, z1 + delta)], [0, z2 - delta]])
            if z2 + delta < 1:
                regions.append(
                    [[max(0, z1 - delta), min(1, z1 + delta)], [z2 + delta, 1]])
            return regions

        result = 0
        for region in get_regions():
            [[x1, x2], [y1, y2]] = region
            integral, _ = integrate.dblquad(
                integrand, y1, y2, lambda y: x1, lambda y: x2,
                epsabs=1e-6, epsrel=1e-6
            )
            result += integral

        def polar_integrand(r, theta):
            x = z[0] + r * np.cos(theta)
            y = z[1] + r * np.sin(theta)
            if not (0 <= x <= 1 and 0 <= y <= 1):
                return 0
            return -1 / (2 * np.pi) * r * f_func(x, y) * np.log(r)

        delta = 0.1
        r_integral, _ = integrate.dblquad(
            polar_integrand, 0, 2 * np.pi,
            lambda theta: eps, lambda theta: delta,
            epsabs=1e-6, epsrel=1e-6
        )

        result += r_integral
        return result

    def compute_Qf_grid(self, f_func):
        """
        Compute Q(f) on a grid of points
        """
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        Qf = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                z = [x[i], y[j]]
                Qf[i, j] = self.get_Qf(f_func, z)
                if (i * self.grid_size + j) % 100 == 0:
                    print(
                        f"Computing Q(f): {(i * self.grid_size + j) / (self.grid_size ** 2) * 100:.1f}% complete")

        return torch.tensor(Qf, dtype=torch.float32)

    def generate_data(self):
        """Generate data using the paper's method"""
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
        """Prepare data with principal symbol decomposition"""
        n_train = int(0.9 * self.data_points)

        # Ensure correct dimensions
        if len(y_data.shape) == 3:
            y_data = y_data.unsqueeze(1)
        if len(x_data.shape) == 3:
            x_data = x_data.unsqueeze(1)

        # Move to device
        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)

        # Compute Q(f) for each input function
        Qf_data = torch.zeros_like(x_data)

        for i in range(len(x_data)):
            # Create interpolation function from grid data
            grid_values = x_data[i, 0].cpu().numpy()
            x_grid = np.linspace(0, 1, self.grid_size)
            y_grid = np.linspace(0, 1, self.grid_size)

            f_interp = RegularGridInterpolator(
                (x_grid, y_grid),
                grid_values,
                method='linear',
                bounds_error=False,
                fill_value=0
            )

            def f_func(x, y):
                return float(f_interp(np.array([x, y])))

            Qf = self.compute_Qf_grid(f_func)
            Qf_data[i, 0] = Qf.to(self.device)
            print(f"Processed Q(f) for function {i + 1}/{len(x_data)}")

        # Compute residuals
        residual_data = y_data - Qf_data

        # Split into train/test
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

    def train_fno(self, x_train, residual_train, Qf_train,
                  x_test, residual_test, Qf_test, n_epochs=20):
        """Train FNO on the residual"""
        model = FNO2d(
            n_modes_height=12,
            n_modes_width=12,
            hidden_channels=32,
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

            # Forward pass on residual
            pred_residual = model(x_train)
            # Full prediction includes principal part
            pred_full = pred_residual + Qf_train
            loss = F.mse_loss(pred_full, y_train)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_pred_residual = model(x_test)
                    test_pred_full = test_pred_residual + Qf_test
                    test_loss = F.mse_loss(test_pred_full, y_test).item()
                    train_loss = loss.item()

                    train_losses.append(train_loss)
                    test_losses.append(test_loss)

                    print(
                        f'Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

        return model, train_losses, test_losses

    def plot_comparison(self, model, x_test, y_test, Qf_test, num_samples=5):
        """Plot comparisons including principal part contribution"""
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
        """Analyze frequency content including principal part contribution"""
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


def main():
    # Initialize tester with smaller grid size for testing
    tester = FNOTesterWithPrincipalSymbol(data_points=100, grid_size=32,
                                          truncation_order=20)

    # Generate/load data
    print("Generating/loading data...")
    x_data, y_data = tester.generate_data()

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


if __name__ == "__main__":
    main()
