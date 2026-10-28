import os
import sys
import json
import argparse
from typing import Dict, Any, Union, List

import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch
import torchaudio

from src.models import AASIST, AASISTLight, AMSoftmaxLoss
from src.utils import load_checkpoint
from src.train import get_device


class DeepfakeDetector:
    """
    Production-ready inference wrapper for AASIST audio anti-spoofing model
    with integrated multi-modal acoustic forensic biometrics.
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        arch: str = "AASIST",
        sample_rate: int = 16000,
        num_samples: int = 64600,
        threshold: float = 0.0,
        device: str = "auto",
    ):
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.threshold = threshold
        self.device = get_device(device)

        if arch == "AASIST-L":
            self.model = AASISTLight(sample_rate=sample_rate).to(self.device)
        else:
            self.model = AASIST(sample_rate=sample_rate).to(self.device)

        self.loss_module = AMSoftmaxLoss(in_features=self.model.embedding_dim, num_classes=2).to(self.device)

        if checkpoint_path and os.path.isfile(checkpoint_path):
            load_checkpoint(checkpoint_path, self.model, loss_module=self.loss_module, device=str(self.device))
            print(f"[*] Loaded checkpoint from: {checkpoint_path}")
        else:
            print("[!] No checkpoint loaded: running with initialized weights (for demonstration).")

        self.model.eval()

    def preprocess_audio(self, audio_path_or_waveform: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Load, resample to 16kHz, convert to mono, and pad/slice to target length."""
        from src.utils.audio_io import robust_load_audio
        if isinstance(audio_path_or_waveform, str):
            waveform, _ = robust_load_audio(audio_path_or_waveform, target_sr=self.sample_rate)
        elif isinstance(audio_path_or_waveform, np.ndarray):
            waveform = torch.from_numpy(audio_path_or_waveform).float()
            if waveform.ndim > 1:
                waveform = torch.mean(waveform, dim=-1)
        else:
            waveform = audio_path_or_waveform.float()
            if waveform.ndim > 1:
                waveform = torch.mean(waveform, dim=0)

        # Pad or center-slice to fixed length
        length = waveform.size(0)
        if length < self.num_samples:
            repeat_times = int(np.ceil(self.num_samples / length))
            waveform = waveform.repeat(repeat_times)[: self.num_samples]
        elif length > self.num_samples:
            start_idx = (length - self.num_samples) // 2
            waveform = waveform[start_idx : start_idx + self.num_samples]

        return waveform.unsqueeze(0).to(self.device) # (1, num_samples)

    def extract_forensic_cues(
        self, audio_path_or_waveform: Union[str, np.ndarray, torch.Tensor]
    ) -> (float, List[str]):
        """
        Extract acoustic forensic artifacts (Nyquist brickwall cutoff,
        formant dynamic rigidity, and vocoder biological inconsistencies).
        """
        from src.utils.audio_io import robust_load_audio
        from src.utils.acoustic_analysis import extract_spectral_biometrics

        forensic_score = 0.0
        artifacts = []

        # 1. Native sample rate frequency bandwidth analysis
        if isinstance(audio_path_or_waveform, str) and os.path.isfile(audio_path_or_waveform):
            try:
                data, orig_sr = sf.read(audio_path_or_waveform)
                if data.ndim > 1:
                    data = np.mean(data, axis=-1)

                f, pxx = signal.welch(data, fs=orig_sr, nperseg=min(2048, len(data)))
                tot_e = np.sum(pxx) + 1e-12

                # Detect brickwall cutoff common in neural vocoders (22.05k / 24k models upsampled to 44.1k/48k)
                if orig_sr >= 32000:
                    e_16k = float(np.sum(pxx[f >= 16000]) / tot_e)
                    e_14k = float(np.sum(pxx[f >= 14000]) / tot_e)
                    e_12k = float(np.sum(pxx[f >= 12000]) / tot_e)

                    if e_16k < 0.000015:
                        forensic_score += 0.45
                        artifacts.append("Nyquist brickwall cutoff at 14–16 kHz (Neural vocoder upsampling artifact)")
                    elif e_14k < 0.0005:
                        forensic_score += 0.25
                        artifacts.append("Abrupt high-frequency suppression above 14 kHz")
            except Exception:
                pass

        # 2. 16kHz resampled articulatory & vocal tract dynamics
        try:
            if isinstance(audio_path_or_waveform, str):
                waveform, sr = robust_load_audio(audio_path_or_waveform, target_sr=16000)
                audio_16k = waveform.cpu().numpy()
            elif isinstance(audio_path_or_waveform, np.ndarray):
                audio_16k = audio_path_or_waveform
            else:
                audio_16k = audio_path_or_waveform.cpu().numpy()

            if audio_16k.ndim > 1:
                audio_16k = np.mean(audio_16k, axis=-1)

            bio = extract_spectral_biometrics(audio_16k, sample_rate=16000)
            comp = bio.get("biometric_comparison", {})

            formant_val = comp.get("formant_transition", {}).get("audio_val", 75)
            tract_val = comp.get("vocal_tract_consistency", {}).get("audio_val", 80)

            # Neural TTS models (VITS, HiFi-GAN, EdgeTTS) generate unnaturally rigid formant transitions
            if formant_val <= 25:
                forensic_score += 0.35
                artifacts.append(f"Unnaturally rigid vocal tract dynamics ({formant_val}/100, typical of neural TTS frame synthesis)")
            elif formant_val <= 45:
                forensic_score += 0.15
                artifacts.append(f"Depressed articulatory movement ({formant_val}/100)")

            # Biological consistency check
            if tract_val <= 62:
                forensic_score += 0.15
                artifacts.append(f"Sub-baseline biological vocal tract consistency ({tract_val}/100)")

        except Exception:
            pass

        return min(0.98, forensic_score), artifacts

    @torch.no_grad()
    def predict(self, audio_path_or_waveform: Union[str, np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Run inference on audio input combining AASIST deep embeddings
        with acoustic forensic biometric evidence.
        """
        tensor = self.preprocess_audio(audio_path_or_waveform)

        embeddings, _ = self.model.forward_features(tensor)
        scores = self.loss_module.get_normalized_scores(embeddings).cpu().numpy()[0]

        # scores[0]: spoof score, scores[1]: bonafide score
        spoof_score = float(scores[0])
        bonafide_score = float(scores[1])
        diff_score = bonafide_score - spoof_score

        # Sigmoid calibration to probability [0, 1]
        neural_spoof_prob = 1.0 / (1.0 + np.exp(diff_score * 3.0))

        # Extract acoustic forensic artifacts
        forensic_spoof_prob, detected_artifacts = self.extract_forensic_cues(audio_path_or_waveform)

        # Multi-modal fusion
        if forensic_spoof_prob >= 0.50:
            # Strong forensic detection (e.g. brickwall cutoff + rigid formants)
            spoof_prob = min(0.985, max(neural_spoof_prob, 0.4 * neural_spoof_prob + 0.6 * forensic_spoof_prob + 0.25))
            final_margin = -round(spoof_prob * 2.2, 4)
        elif forensic_spoof_prob >= 0.25:
            spoof_prob = 0.5 * neural_spoof_prob + 0.5 * forensic_spoof_prob
            final_margin = round(diff_score - (forensic_spoof_prob * 1.5), 4)
        else:
            spoof_prob = neural_spoof_prob
            final_margin = round(float(diff_score), 4)

        bonafide_prob = 1.0 - spoof_prob
        is_spoof = spoof_prob >= 0.50 or diff_score < self.threshold
        verdict = "SPOOF / AI VOICE DETECTED" if is_spoof else "BONAFIDE (GENUINE HUMAN)"
        confidence = float(spoof_prob if is_spoof else bonafide_prob) * 100.0

        if spoof_prob > 0.80:
            risk_level = "CRITICAL - HIGH LIKELIHOOD OF SYNTHESIS (AI VOICE DETECTED)"
        elif spoof_prob > 0.50:
            risk_level = "SUSPICIOUS - POTENTIAL VOCODER / TTS ARTIFACTS"
        elif spoof_prob > 0.20:
            risk_level = "LOW RISK - LIKELY NATURAL SPEECH"
        else:
            risk_level = "VERY LOW RISK - HIGH CONFIDENCE HUMAN"

        return {
            "verdict": verdict,
            "is_spoof": bool(is_spoof),
            "spoof_probability": round(float(spoof_prob), 4),
            "bonafide_probability": round(float(bonafide_prob), 4),
            "confidence_pct": round(confidence, 2),
            "bonafide_raw_score": round(bonafide_score, 4),
            "spoof_raw_score": round(spoof_score, 4),
            "decision_margin": final_margin,
            "risk_level": risk_level,
            "detected_artifacts": detected_artifacts,
        }


def main():
    parser = argparse.ArgumentParser(description="Predict deepfake audio using AASIST")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Decision threshold")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    detector = DeepfakeDetector(checkpoint_path=args.checkpoint, threshold=args.threshold)
    result = detector.predict(args.audio)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*55}")
        print("🎙️ AUDIO DEEPFAKE DETECTION RESULT")
        print(f"{'='*55}")
        print(f"File                  : {os.path.basename(args.audio)}")
        print(f"Classification Verdict: {result['verdict']}")
        print(f"Confidence Level      : {result['confidence_pct']:.2f}%")
        print(f"Spoof Probability     : {result['spoof_probability'] * 100:.2f}%")
        print(f"Bona fide Probability : {result['bonafide_probability'] * 100:.2f}%")
        print(f"Raw Decision Margin   : {result['decision_margin']:+.4f}")
        print(f"Security Assessment   : {result['risk_level']}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
