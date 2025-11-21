"""
Spectogram Audio Augmentation After Wave Augmentation on a HuggingFace dataset's audio column 
Author: Ye Bhone Lin
Date: 21st Nov 2025
"""

import os
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np

def time_mask(mel, width=30):
    aug = mel.copy()
    t = np.random.randint(0, width)
    t0 = np.random.randint(0, max(1, aug.shape[1]-t))
    aug[:, t0:t0+t] = 0
    return aug

def freq_mask(mel, width=15):
    aug = mel.copy()
    f = np.random.randint(0, width)
    f0 = np.random.randint(0, max(1, aug.shape[0]-f))
    aug[f0:f0+f, :] = 0
    return aug

class SpectrogramAugmentationPipeline:
    def __init__(self):
        self.methods = {
            'time_mask': time_mask,
            'freq_mask': freq_mask
        }
    def augment(self, mel, method):
        if method not in self.methods:
            raise ValueError(f"Unknown spec method: {method}")
        return self.methods[method](mel)

class SpecAugmentationAfterWav:
    def __init__(self, base_input_dir, base_output_dir, sr=16000, n_mels=80):
        self.base_input_dir = Path(base_input_dir)
        self.base_output_dir = Path(base_output_dir)
        self.sr = sr
        self.n_mels = n_mels
        self.pipeline = SpectrogramAugmentationPipeline()

    def augment(self, spec_methods):
        for folder in self.base_input_dir.iterdir():
            if not folder.is_dir():
                continue

            wav_files = list(folder.glob("*.wav"))
            if not wav_files:
                continue

            for wav_path in wav_files:
                y, sr = librosa.load(wav_path, sr=self.sr)
                mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
                log_mel_spec = librosa.power_to_db(mel_spec)

                for method in spec_methods:
                    aug_spec = self.pipeline.augment(log_mel_spec, method)
                    mel_spec_recon = librosa.db_to_power(aug_spec)
                    y_recon = librosa.feature.inverse.mel_to_audio(mel_spec_recon, sr=self.sr)

                    out_dir = self.base_output_dir / folder.name / method
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / wav_path.name
                    sf.write(out_path, y_recon, self.sr, format="WAV", subtype="PCM_16")
                    print(f"Saved: {out_path}")

#if __name__ == "__main__":
#    base_input_dir = "output_of_aug2"  # folder containing speed/, pitch/, etc.
#    base_output_dir = "spec_augmented"

#    spec_methods = ['time_mask', 'freq_mask']  # you can add 'spec_augment' if needed

#    pipeline = SpecAugmentationAfterWav(base_input_dir, base_output_dir)
#    pipeline.augment(spec_methods)
