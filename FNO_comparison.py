"""
Comparison of standard FNO with FNO + Principal Symbol decomposition
"""
from test_FNO_pseudo_ifft import FNOTesterWithPrincipalSymbol
from test_FNO import FNOTester
import torch
import matplotlib.pyplot as plt
import numpy as np


def compare_methods(data_points=150, grid_size=64, truncation_order=10,
                    num_samples=5):
    # Initialize both testers with identical parameters
    tester_ps = FNOTesterWithPrincipalSymbol(data_points, grid_size,
                                             truncation_order)
    tester_std = FNOTester(data_points, grid_size, truncation_order)

    # Generate data once and share between methods
    print("Generating/loading data...")
    x_data, y_data = tester_ps.generate_data()

    # Prepare data for both methods
    print("\nPreparing data...")
    # Principal Symbol method
    (x_train_ps, y_train_ps, Qf_train, residual_train,
     x_test_ps, y_test_ps, Qf_test, residual_test) = tester_ps.prepare_data(
        x_data, y_data)

    # Standard FNO method
    x_train_std, y_train_std, x_test_std, y_test_std = tester_std.prepare_data(
        x_data, y_data)

    # Train both models
    print("\nTraining Principal Symbol FNO...")
    model_ps, train_losses_ps, test_losses_ps = tester_ps.train_fno(
        x_train_ps, residual_train, Qf_train,
        x_test_ps, residual_test, Qf_test
    )

    print("\nTraining Standard FNO...")
    model_std, train_losses_std, test_losses_std = tester_std.train_fno(
        x_train_std, y_train_std, x_test_std, y_test_std
    )

    # Compare results on same test samples
    indices = np.random.choice(len(x_test_ps), num_samples, replace=False)

    for i, idx in enumerate(indices):
        # Get predictions from both methods
        with torch.no_grad():
            # Principal Symbol method
            pred_residual_ps = model_ps(x_test_ps[idx:idx + 1])
            pred_full_ps = pred_residual_ps + Qf_test[idx:idx + 1]

            # Standard FNO method
            pred_std = model_std(x_test_std[idx:idx + 1])

        # Plot comparisons
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Sample {i + 1} Comparison', fontsize=16)

        # Input function
        im1 = axes[0, 0].imshow(x_test_ps[idx, 0].cpu(), cmap='viridis')
        axes[0, 0].set_title('Input f')
        plt.colorbar(im1, ax=axes[0, 0])

        # True solution
        im2 = axes[0, 1].imshow(y_test_ps[idx, 0].cpu(), cmap='viridis')
        axes[0, 1].set_title('True Solution u')
        plt.colorbar(im2, ax=axes[0, 1])

        # Principal Symbol prediction
        im3 = axes[0, 2].imshow(pred_full_ps[0, 0].cpu(), cmap='viridis')
        axes[0, 2].set_title('$\Psi$-FNO Prediction')
        plt.colorbar(im3, ax=axes[0, 2])

        # Standard FNO prediction
        im4 = axes[1, 0].imshow(pred_std[0, 0].cpu(), cmap='viridis')
        axes[1, 0].set_title('Standard FNO Prediction')
        plt.colorbar(im4, ax=axes[1, 0])

        # Error maps
        error_ps = torch.abs(pred_full_ps[0, 0] - y_test_ps[idx, 0]).cpu()
        error_std = torch.abs(pred_std[0, 0] - y_test_std[idx, 0]).cpu()

        im5 = axes[1, 1].imshow(error_ps, cmap='hot')
        axes[1, 1].set_title('$\Psi$-FNO Error Map')
        plt.colorbar(im5, ax=axes[1, 1])

        im6 = axes[1, 2].imshow(error_std, cmap='hot')
        axes[1, 2].set_title('Standard FNO Error Map')
        plt.colorbar(im6, ax=axes[1, 2])

        plt.tight_layout()
        plt.show()

        # Print error metrics
        ps_error = torch.norm(pred_full_ps[0] - y_test_ps[idx]) / torch.norm(
            y_test_ps[idx])
        std_error = torch.norm(pred_std[0] - y_test_std[idx]) / torch.norm(
            y_test_std[idx])

        print(f'\nSample {i + 1} Metrics:')
        print(f'PS-FNO Relative L2 Error: {ps_error:.6f}')
        print(f'Standard FNO Relative L2 Error: {std_error:.6f}')
        print(
            f'Error Reduction: {((std_error - ps_error) / std_error * 100):.2f}%')

        # Frequency analysis
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Sample {i + 1} Frequency Content', fontsize=16)

        # True solution frequencies
        fft_true = torch.fft.fft2(y_test_ps[idx, 0].cpu())
        mag_true = torch.abs(fft_true)
        im1 = ax1.imshow(torch.log(mag_true + 1e-10), cmap='viridis')
        ax1.set_title('True Solution')
        plt.colorbar(im1, ax=ax1)

        # PS-FNO frequencies
        fft_ps = torch.fft.fft2(pred_full_ps[0, 0].cpu())
        mag_ps = torch.abs(fft_ps)
        im2 = ax2.imshow(torch.log(mag_ps + 1e-10), cmap='viridis')
        ax2.set_title('$\Psi$-FNO Prediction')
        plt.colorbar(im2, ax=ax2)

        # Standard FNO frequencies
        fft_std = torch.fft.fft2(pred_std[0, 0].cpu())
        mag_std = torch.abs(fft_std)
        im3 = ax3.imshow(torch.log(mag_std + 1e-10), cmap='viridis')
        ax3.set_title('Standard FNO Prediction')
        plt.colorbar(im3, ax=ax3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    compare_methods()