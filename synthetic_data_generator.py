import os
import argparse
import random
import numpy as np
import soundfile as sf


def generate_synthetic_audio(duration_sec: float = 4.0, sr: int = 16000, is_spoof: bool = False) -> np.ndarray:
    """Generate audio signal simulating either natural speech or synthetic vocoder audio."""
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    if not is_spoof:
        # Simulate natural speech formants: fundamental f0 + formant harmonics with vibrato
        f0 = random.uniform(120, 220)
        vibrato = 1.0 + 0.03 * np.sin(2 * np.pi * 5.0 * t)
        signal = 0.5 * np.sin(2 * np.pi * f0 * vibrato * t)
        # Formants
        signal += 0.3 * np.sin(2 * np.pi * (2 * f0) * vibrato * t)
        signal += 0.2 * np.sin(2 * np.pi * (3 * f0) * vibrato * t)
        signal += 0.1 * np.sin(2 * np.pi * (4 * f0) * vibrato * t)
        # Add smooth envelope
        envelope = 0.5 * (1.0 - np.cos(2 * np.pi * np.linspace(0, 1, num_samples)))
        signal = signal * envelope
        # Low natural background noise
        noise = np.random.normal(0, 0.005, num_samples)
        signal = signal + noise
    else:
        # Simulate synthetic vocoder: phase discontinuity, buzz, robotic buzz, or high-freq artifacts
        f0 = random.uniform(140, 260)
        # Harsh square / sawtooth harmonic comb
        signal = 0.4 * np.sign(np.sin(2 * np.pi * f0 * t))
        # Robot modulation
        signal *= (1.0 + 0.5 * np.sin(2 * np.pi * 30.0 * t))
        # High-frequency vocoder hiss
        hiss = np.random.normal(0, 0.05, num_samples)
        signal = signal + hiss

    # Normalize to [-0.95, 0.95]
    max_val = np.max(np.abs(signal)) + 1e-6
    signal = 0.95 * signal / max_val
    return signal.astype(np.float32)


def generate_mock_dataset(output_dir: str = "./data", num_samples: int = 40):
    """
    Generate mock folder structure and protocol files for ASVspoof 2019 & 2021.
    """
    sr = 16000

    # 1. ASVspoof 2019 LA Train
    la_19_train_dir = os.path.join(output_dir, "asvspoof2019", "LA", "ASVspoof2019_LA_train", "flac")
    la_19_train_proto_dir = os.path.join(output_dir, "asvspoof2019", "LA", "ASVspoof2019_LA_cm_protocols")
    os.makedirs(la_19_train_dir, exist_ok=True)
    os.makedirs(la_19_train_proto_dir, exist_ok=True)
    la_19_train_proto = os.path.join(la_19_train_proto_dir, "ASVspoof2019.LA.cm.train.trn.txt")

    # 2. ASVspoof 2019 LA Dev
    la_19_dev_dir = os.path.join(output_dir, "asvspoof2019", "LA", "ASVspoof2019_LA_dev", "flac")
    la_19_dev_proto_dir = la_19_train_proto_dir
    os.makedirs(la_19_dev_dir, exist_ok=True)
    la_19_dev_proto = os.path.join(la_19_dev_proto_dir, "ASVspoof2019.LA.cm.dev.trl.txt")

    # 3. ASVspoof 2021 LA Eval
    la_21_eval_dir = os.path.join(output_dir, "asvspoof2021", "LA", "ASVspoof2021_LA_eval", "flac")
    la_21_eval_keys_dir = os.path.join(output_dir, "asvspoof2021", "LA", "ASVspoof2021_LA_eval", "keys", "LA", "CM")
    os.makedirs(la_21_eval_dir, exist_ok=True)
    os.makedirs(la_21_eval_keys_dir, exist_ok=True)
    la_21_eval_proto = os.path.join(la_21_eval_keys_dir, "trial_metadata.txt")

    attacks = [f"A{i:02d}" for i in range(1, 20)]

    print(f"[*] Generating {num_samples} mock audio files and protocols...")

    # Train files
    with open(la_19_train_proto, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            is_spoof = (i % 2 == 0)
            utt_id = f"LA_T_{1000000 + i}"
            spk_id = f"LA_{100 + (i % 5)}"
            key = "spoof" if is_spoof else "bonafide"
            atk = random.choice(attacks) if is_spoof else "-"

            audio = generate_synthetic_audio(duration_sec=random.uniform(2.5, 4.5), sr=sr, is_spoof=is_spoof)
            sf.write(os.path.join(la_19_train_dir, f"{utt_id}.flac"), audio, sr)

            f.write(f"{spk_id} {utt_id} - {atk} {key}\n")

    # Dev files
    with open(la_19_dev_proto, "w", encoding="utf-8") as f:
        for i in range(num_samples // 2):
            is_spoof = (i % 2 == 0)
            utt_id = f"LA_D_{2000000 + i}"
            spk_id = f"LA_{200 + (i % 5)}"
            key = "spoof" if is_spoof else "bonafide"
            atk = random.choice(attacks) if is_spoof else "-"

            audio = generate_synthetic_audio(duration_sec=random.uniform(2.5, 4.5), sr=sr, is_spoof=is_spoof)
            sf.write(os.path.join(la_19_dev_dir, f"{utt_id}.flac"), audio, sr)

            f.write(f"{spk_id} {utt_id} - {atk} {key}\n")

    # 2021 Eval files
    with open(la_21_eval_proto, "w", encoding="utf-8") as f:
        for i in range(num_samples // 2):
            is_spoof = (i % 2 == 1)
            utt_id = f"LA_E_{3000000 + i}"
            spk_id = f"LA_{300 + (i % 5)}"
            key = "spoof" if is_spoof else "bonafide"
            atk = random.choice(attacks) if is_spoof else "-"

            audio = generate_synthetic_audio(duration_sec=random.uniform(2.5, 4.5), sr=sr, is_spoof=is_spoof)
            sf.write(os.path.join(la_21_eval_dir, f"{utt_id}.flac"), audio, sr)

            f.write(f"{spk_id} {utt_id} alaw g722 ip {atk} {key} eval\n")

    print("[✔] Mock datasets generated successfully!")
    print(f"    Train manifest: {la_19_train_proto}")
    print(f"    Dev manifest  : {la_19_dev_proto}")
    print(f"    2021 Eval keys: {la_21_eval_proto}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ASVspoof data for testing")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of samples to generate")
    args = parser.parse_args()

    generate_mock_dataset(args.output_dir, args.num_samples)
