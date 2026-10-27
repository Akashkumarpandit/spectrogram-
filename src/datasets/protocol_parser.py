import os
import glob
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class AudioProtocolItem:
    utt_id: str
    audio_path: str
    label_str: str # "bonafide" or "spoof"
    label: int     # 1: bonafide, 0: spoof
    attack_type: str
    speaker_id: Optional[str] = None
    dataset_version: str = "2019"


def find_audio_file(audio_dir: str, utt_id: str) -> Optional[str]:
    """Search for the audio file matching utt_id in audio_dir with various extensions."""
    # 1. Exact match
    direct_path = os.path.join(audio_dir, utt_id)
    if os.path.isfile(direct_path):
        return direct_path

    # 2. Extensions: .flac, .wav, .mp3, .m4a
    for ext in [".flac", ".wav", ".mp3", ".m4a"]:
        path = os.path.join(audio_dir, f"{utt_id}{ext}")
        if os.path.isfile(path):
            return path

    # 3. If utt_id already has an extension but doesn't exist directly
    base, _ = os.path.splitext(utt_id)
    for ext in [".flac", ".wav", ".mp3"]:
        path = os.path.join(audio_dir, f"{base}{ext}")
        if os.path.isfile(path):
            return path

    return None


def parse_protocol_file(
    protocol_path: str,
    audio_dir: str,
    dataset_version: str = "2019",
    verify_audio_exists: bool = False,
) -> List[AudioProtocolItem]:
    """
    Parse ASVspoof 2019 or ASVspoof 2021 protocol / metadata files.
    
    ASVspoof 2019 format (space delimited):
      [SpeakerID] [AudioID] [Env] [AttackType] [Key]
      e.g.: LA_0079 LA_T_1138215 - - bonafide
      e.g.: LA_0079 LA_T_1271820 - A01 spoof

    ASVspoof 2021 trial_metadata format (space delimited):
      LA: [SpeakerID] [AudioID] [Codec] [Channel] [Env] [AttackType] [Key] ...
      DF: [AudioID] [SrcDataset] [Codec] [Channel] ... [Key]
    """
    if not os.path.isfile(protocol_path):
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    items: List[AudioProtocolItem] = []

    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.split()
            if len(tokens) < 3:
                continue

            # Determine key (label) and utt_id
            # In ASVspoof, the key 'bonafide' or 'spoof' is almost always the 5th token (2019) or last token / keyword
            label_str = None
            utt_id = None
            attack_type = "unknown"
            speaker_id = None

            # Detect label from tokens
            lower_tokens = [t.lower() for t in tokens]
            if "bonafide" in lower_tokens:
                label_idx = lower_tokens.index("bonafide")
                label_str = "bonafide"
            elif "spoof" in lower_tokens:
                label_idx = lower_tokens.index("spoof")
                label_str = "spoof"
            else:
                # Could be integer 1/0 or custom
                if tokens[-1] in ["1", "0"]:
                    label_str = "bonafide" if tokens[-1] == "1" else "spoof"
                    label_idx = len(tokens) - 1
                else:
                    continue # Skip unparseable line

            label = 1 if label_str == "bonafide" else 0

            # Determine utt_id and metadata based on layout
            if dataset_version == "2019":
                # Standard 2019 layout:
                # SpeakerID (0), AudioID (1), Env (2), AttackType (3), Key (4)
                speaker_id = tokens[0]
                utt_id = tokens[1]
                attack_type = tokens[3] if len(tokens) > 3 else "-"
            elif dataset_version == "2021":
                # 2021 layout:
                if tokens[0].startswith("LA_") or tokens[0].startswith("PA_"):
                    speaker_id = tokens[0]
                    utt_id = tokens[1]
                else:
                    utt_id = tokens[0] # DF often has AudioID first
                attack_type = tokens[label_idx - 1] if label_idx > 0 else "-"
            else:
                utt_id = tokens[1] if len(tokens) > 1 else tokens[0]

            audio_path = find_audio_file(audio_dir, utt_id)
            if verify_audio_exists and audio_path is None:
                continue
            elif audio_path is None:
                # Fallback path if not verifying existence right now
                audio_path = os.path.join(audio_dir, f"{utt_id}.flac")

            items.append(
                AudioProtocolItem(
                    utt_id=utt_id,
                    audio_path=audio_path,
                    label_str=label_str,
                    label=label,
                    attack_type=attack_type,
                    speaker_id=speaker_id,
                    dataset_version=dataset_version,
                )
            )

    return items
