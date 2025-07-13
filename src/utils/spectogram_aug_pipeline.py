import csv
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import librosa
import cv2
from scipy.signal import istft

# === Spectrogram Augmentation Functions ===
def time_mask(mel, width=30):
    aug = mel.copy()
    t = np.random.randint(0, width)
    t0 = np.random.randint(0, max(1, aug.shape[1] - t))
    aug[:, t0:t0+t] = 0
    return aug

def freq_mask(mel, width=15):
    aug = mel.copy()
    f = np.random.randint(0, width)
    f0 = np.random.randint(0, max(1, aug.shape[0] - f))
    aug[f0:f0+f, :] = 0
    return aug

def spec_augment(mel, time_mask_width=30, freq_mask_width=15):
    return freq_mask(time_mask(mel, time_mask_width), freq_mask_width)

def time_warp(mel, max_warp=5):
    import scipy.ndimage
    return scipy.ndimage.shift(mel, shift=(0, np.random.randint(-max_warp, max_warp)), mode='nearest')

def add_noise(mel, level=0.01):
    return mel + np.random.randn(*mel.shape) * level

def time_shift(mel, max_shift=10):
    return np.roll(mel, np.random.randint(-max_shift, max_shift), axis=1)

def freq_shift(mel, max_shift=5):
    return np.roll(mel, np.random.randint(-max_shift, max_shift), axis=0)

def resize_crop(mel, scale_range=(0.8, 1.2)):
    num_mels, num_frames = mel.shape
    scale = np.random.uniform(*scale_range)
    new_mels, new_frames = int(num_mels * scale), int(num_frames * scale)
    resized = cv2.resize(mel, (new_frames, new_mels), interpolation=cv2.INTER_LINEAR)

    if scale >= 1.0:
        start_m = (new_mels - num_mels) // 2
        start_f = (new_frames - num_frames) // 2
        return resized[start_m:start_m+num_mels, start_f:start_f+num_frames]
    else:
        pad_m = (num_mels - new_mels) // 2
        pad_f = (num_frames - new_frames) // 2
        return np.pad(resized, ((pad_m, num_mels - new_mels - pad_m), (pad_f, num_frames - new_frames - pad_f)))

def dynamic_range_compression(mel, C=1, clip_val=1e-5):
    return np.log10(C * np.maximum(mel, clip_val))

def band_drop(mel, prob=0.1):
    mask = np.random.rand(mel.shape[0]) < prob
    mel[mask, :] = 0
    return mel

def patch_swap(mel, patch_size=(10, 10)):
    h, w = mel.shape
    ph, pw = patch_size
    if h < ph or w < pw:
        return mel
    m1, n1 = np.random.randint(0, h - ph), np.random.randint(0, w - pw)
    m2, n2 = np.random.randint(0, h - ph), np.random.randint(0, w - pw)
    mel[m1:m1+ph, n1:n1+pw], mel[m2:m2+ph, n2:n2+pw] = mel[m2:m2+ph, n2:n2+pw].copy(), mel[m1:m1+ph, n1:n1+pw].copy()
    return mel


# === Spectrogram Augmentation Pipeline ===
class SpectrogramAugmentationPipeline:
    def __init__(self, sr):
        self.sr = sr
        self.methods = {
            'time_mask': time_mask,
            'freq_mask': freq_mask,
            'spec_augment': spec_augment,
            'time_warp': time_warp,
            'add_noise': add_noise,
            'time_shift': time_shift,
            'freq_shift': freq_shift,
            'resize_crop': resize_crop,
            'dynamic_range_compression': dynamic_range_compression,
            'band_drop': band_drop,
            'patch_swap': patch_swap,
        }

    def augment(self, mel_spec, method):
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        return self.methods[method](mel_spec)


class SpectrogramAugmentation:
    def __init__(self, metadata_path, output_dir, sr=16000, n_mels=80):
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.n_mels = n_mels
        self.pipeline = SpectrogramAugmentationPipeline(sr=sr)

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
                    wav_path_str, text = line.split("|")
                    wav_path = Path(wav_path_str).resolve()

                    if not wav_path.exists():
                        print(f"Missing file: {wav_path}")
                        continue

                    y, sr = librosa.load(wav_path, sr=self.sr)
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
                    log_mel_spec = librosa.power_to_db(mel_spec)

                    aug_spec = self.pipeline.augment(log_mel_spec, method)

                    # Inverse mel and save
                    mel_spec_recon = librosa.db_to_power(aug_spec)
                    y_recon = librosa.feature.inverse.mel_to_audio(mel_spec_recon, sr=self.sr)

                    out_filename = f"{method}_{wav_path.name}"
                    out_path = method_dir / out_filename
                    sf.write(out_path, y_recon, self.sr, format="WAV", subtype="PCM_16")

                    meta_out.write(f"{out_path.resolve()}|{text.strip()}\n")
                    print(f"Saved augmented: {out_path}")


class HF_SpectrogramAugmentation:
    def __init__(self, metadata_path, output_dir, sr=16000, n_mels=80):
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.n_mels = n_mels
        self.pipeline = SpectrogramAugmentationPipeline(sr=sr)

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
                        print(f"Missing file: {wav_path}")
                        continue

                    y, sr = librosa.load(wav_path, sr=self.sr)
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
                    log_mel_spec = librosa.power_to_db(mel_spec)

                    aug_spec = self.pipeline.augment(log_mel_spec, method)

                    # Inverse mel and save
                    mel_spec_recon = librosa.db_to_power(aug_spec)
                    y_recon = librosa.feature.inverse.mel_to_audio(mel_spec_recon, sr=self.sr)

                    out_filename = f"{method}_{wav_path.name}"
                    out_path = method_dir / out_filename
                    sf.write(out_path, y_recon, self.sr, format="WAV", subtype="PCM_16")

                    meta_out.write(f"{out_path.resolve()}\t{text.strip()}\n")
                    print(f"Saved augmented: {out_path}")

