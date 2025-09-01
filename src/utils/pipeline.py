"""
Audio  Augmentation on a HuggingFace dataset's audio column & Local Dataset 
Author: Ye Bhone Lin
Date: 1st Sep 2025
"""


import csv
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import nlpaug.augmenter.audio as naa


class AudioAugmentationPipeline:
    def __init__(self, sr):
        self.sr = sr
        self.augmenters = {
            'loudness': naa.LoudnessAug(),
            'crop': naa.CropAug(sampling_rate=sr),
            'mask': naa.MaskAug(sampling_rate=sr, zone=(0.0, 1.0), coverage=0.1, mask_with_noise=False, stateless=True),
            'noise': naa.NoiseAug(),
            'pitch': naa.PitchAug(sampling_rate=sr, factor=(2, 3)),
            'shift': naa.ShiftAug(sampling_rate=sr),
            'speed': naa.SpeedAug(zone=(0.0, 1.0), coverage=1.0, factor=(1.5, 1.5)),
            'vtlp': naa.VtlpAug(sampling_rate=sr),
        }

    def augment(self, data, augmenter_name):
        if augmenter_name not in self.augmenters:
            raise ValueError(f"Augmentation '{augmenter_name}' is not supported.")
        if len(data) < 100:
            raise ValueError("Audio too short.")
        augmented = self.augmenters[augmenter_name].augment(data)
        return np.array(augmented, dtype=np.float32)


class Augmentation:
    def __init__(self, metadata_path, output_dir, sr=16000):
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.pipeline = AudioAugmentationPipeline(sr=sr)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.entries = [line.strip() for line in f if "|" in line]

    def augment(self, percent, methods):
        total = len(self.entries)
        sample_count = max(1, int(total * percent / 100))
        selected = random.sample(self.entries, sample_count)

        aug_meta_path = self.output_dir / "aug_metadata.txt"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(aug_meta_path, "w", encoding="utf-8") as meta_out:
            for method in methods:
                method_dir = self.output_dir / method
                method_dir.mkdir(exist_ok=True)

                for line in selected:
                    wav_path, text = line.split("|")
                    wav_path = Path(wav_path).resolve()

                    if not wav_path.exists():
                        print(f"Warning: file not found: {wav_path}")
                        continue

                    # Read audio
                    data, sr = sf.read(wav_path)
                    if sr != self.sr:
                        print(f"Warning: sample rate mismatch for {wav_path}: expected {self.sr}, got {sr}")

                    # Augment audio
                    augmented = self.pipeline.augment(data, method)

                    # Fix shape for soundfile.write
                    if augmented.ndim == 2:
                        augmented = augmented.T
                        if augmented.shape[1] == 1:
                            augmented = augmented.squeeze()

                    # Normalize audio to avoid silence
                    max_val = np.max(np.abs(augmented))
                    if max_val > 0:
                        augmented = augmented / max_val * 0.8

                    # Save new file
                    out_filename = f"{method}_{wav_path.name}"
                    out_path = method_dir / out_filename
                    sf.write(out_path, augmented.astype(np.float32), self.sr, format="WAV", subtype="PCM_16")

                    # Write to new metadata
                    meta_out.write(f"{out_path.resolve()}|{text.strip()}\n")

                    print(f"Augmented: {out_path}")

class HF_Augmentation:
    def __init__(self, metadata_path, output_dir, sr=16000):
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.pipeline = AudioAugmentationPipeline(sr=sr)

        with open(metadata_path, "r", encoding="utf-8", newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            self.entries = [row for row in reader if len(row) >= 2]

    def augment(self, percent, methods):
        total = len(self.entries)
        sample_count = max(1, int(total * percent / 100))
        selected = random.sample(self.entries, sample_count)

        aug_meta_path = self.output_dir / "aug_metadata.txt"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(aug_meta_path, "w", encoding="utf-8") as meta_out:
            for method in methods:
                method_dir = self.output_dir / method
                method_dir.mkdir(exist_ok=True)

                for wav_path_str, text in selected:
                    wav_path = Path(wav_path_str).resolve()

                    if not wav_path.exists():
                        print(f"Warning: file not found: {wav_path}")
                        continue

                    data, sr = sf.read(wav_path)
                    if sr != self.sr:
                        print(f"Warning: sample rate mismatch for {wav_path}: expected {self.sr}, got {sr}")

                    augmented = self.pipeline.augment(data, method)

                    if augmented.ndim == 2:
                        augmented = augmented.T
                        if augmented.shape[1] == 1:
                            augmented = augmented.squeeze()

                    max_val = np.max(np.abs(augmented))
                    if max_val > 0:
                        augmented = augmented / max_val * 0.8

                    out_filename = f"{method}_{wav_path.name}"
                    out_path = method_dir / out_filename
                    sf.write(out_path, augmented.astype(np.float32), self.sr, format="WAV", subtype="PCM_16")

                    meta_out.write(f"{out_path.resolve()}\t{text.strip()}\n")

                    print(f"Augmented: {out_path}")

class SingleAugHF:
    def __init__(self, metadata_path, output_dir, sr=16000):
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.pipeline = AudioAugmentationPipeline(sr=sr)

        with open(metadata_path, "r", encoding="utf-8", newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            self.entries = [row for row in reader if len(row) >= 2]

    def augment_single(self, audio_path_str, method, text=""):
        """
        Augment a single audio file using the specified method.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        method_dir = self.output_dir / method
        method_dir.mkdir(exist_ok=True)

        wav_path = Path(audio_path_str).resolve()
        if not wav_path.exists():
            print(f"Warning: file not found: {wav_path}")
            return

        # Load audio
        data, sr = sf.read(wav_path)
        if sr != self.sr:
            print(f"Warning: sample rate mismatch for {wav_path}: expected {self.sr}, got {sr}")

        # Apply augmentation
        augmented = self.pipeline.augment(data, method)

        if augmented.ndim == 2:
            augmented = augmented.T
            if augmented.shape[1] == 1:
                augmented = augmented.squeeze()

        max_val = np.max(np.abs(augmented))
        if max_val > 0:
            augmented = augmented / max_val * 0.8

        out_filename = f"{method}_{wav_path.name}"
        out_path = method_dir / out_filename
        sf.write(out_path, augmented.astype(np.float32), self.sr, format="WAV", subtype="PCM_16")

        if text:
            aug_meta_path = self.output_dir / "aug_metadata.txt"
            with open(aug_meta_path, "a", encoding="utf-8") as f:
                f.write(f"{out_path.resolve()}\t{text.strip()}\n")

        print(f"Augmented: {out_path}")
