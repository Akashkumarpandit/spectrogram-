import random
import numpy as np
import torch
import scipy.signal as signal


class RawBoost:
    """
    RawBoost: Data augmentation for raw waveform audio anti-spoofing.
    Simulates linear and non-linear filtering, stationary noise, and impulsive noise.
    
    Reference:
        Tak et al., "RawBoost: Someone-to-Someone Data Augmentation for Voice Anti-Spoofing", ICASSP 2022.
    """

    def __init__(self, algo: int = 5, snr_min: float = 10.0, snr_max: float = 40.0):
        self.algo = algo
        self.snr_min = snr_min
        self.snr_max = snr_max

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw waveform tensor of shape (N,) or (1, N)
        Returns:
            Augmented raw waveform tensor
        """
        orig_shape = x.shape
        x_np = x.squeeze().cpu().numpy().astype(np.float32)

        if self.algo == 1:
            x_aug = self._linear_nonlinear_filter(x_np)
        elif self.algo == 2:
            x_aug = self._impulsive_noise(x_np)
        elif self.algo == 3:
            x_aug = self._colored_noise(x_np)
        elif self.algo == 4: # Series combination
            x_aug = self._linear_nonlinear_filter(x_np)
            x_aug = self._impulsive_noise(x_aug)
            x_aug = self._colored_noise(x_aug)
        elif self.algo == 5: # Random selection
            choice = random.choice([1, 2, 3])
            if choice == 1:
                x_aug = self._linear_nonlinear_filter(x_np)
            elif choice == 2:
                x_aug = self._impulsive_noise(x_np)
            else:
                x_aug = self._colored_noise(x_np)
        else:
            x_aug = x_np

        out = torch.from_numpy(x_aug).to(dtype=x.dtype, device=x.device)
        return out.view(orig_shape)

    def _linear_nonlinear_filter(self, x: np.ndarray) -> np.ndarray:
        """Linear FIR filter with mild non-linear saturation."""
        # Random FIR lowpass / bandpass filter
        num_taps = random.randint(5, 25)
        cutoff = random.uniform(0.1, 0.9)
        b = signal.firwin(num_taps, cutoff)
        filtered = signal.lfilter(b, 1.0, x)

        # Mild non-linear saturation
        alpha = random.uniform(0.8, 1.2)
        out = np.tanh(alpha * filtered)
        return out.astype(np.float32)

    def _impulsive_noise(self, x: np.ndarray) -> np.ndarray:
        """Inject random impulsive signal spikes."""
        num_spikes = random.randint(1, 10)
        indices = np.random.randint(0, len(x), size=num_spikes)
        x_noisy = x.copy()
        max_val = np.max(np.abs(x)) + 1e-6
        for idx in indices:
            spike = random.uniform(-max_val, max_val)
            x_noisy[idx] = spike
        return x_noisy.astype(np.float32)

    def _colored_noise(self, x: np.ndarray) -> np.ndarray:
        """Add colored/white noise with random SNR."""
        snr = random.uniform(self.snr_min, self.snr_max)
        sig_power = np.mean(x ** 2) + 1e-8
        noise_power = sig_power / (10 ** (snr / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), size=len(x))
        return (x + noise).astype(np.float32)
