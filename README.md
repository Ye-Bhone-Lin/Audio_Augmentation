# Audio Augmentation Pipeline

This project provides a flexible and extensible pipeline for augmenting audio datasets, especially for Automatic Speech Recognition (ASR) tasks. It supports waveform and spectrogram augmentations, works with both local metadata and HuggingFace datasets, and outputs both augmented audio and updated metadata.

---

## Features

- **Waveform Augmentation:** Add noise, pitch shift, speed change, loudness, crop, mask, shift, VTLP, etc.
- **Spectrogram Augmentation:** Time masking, frequency masking, and more.
- **Flexible Input:** Supports both local metadata files (`metadata.txt`) and HuggingFace-style TSV metadata.
- **Random Sampling:** Augment a random percentage of your dataset.
- **Easy Output:** Augmented audio and new metadata are saved in organized folders.

---

## Augmentation Techniques

### Waveform Augmentation Methods

| Method   | Description                                |
| -------- | ------------------------------------------ |
| loudness | Randomly changes the loudness of the audio |
| crop     | Randomly crops a segment from the audio    |
| mask     | Masks a portion of the audio with silence  |
| noise    | Adds random noise to the audio             |
| pitch    | Shifts the pitch of the audio              |
| shift    | Shifts the audio in time                   |
| speed    | Changes the speed of the audio             |
| vtlp     | Applies Vocal Tract Length Perturbation    |

### Spectrogram Augmentation Methods

| Method                    | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| time_mask                 | Masks a random range of time frames in the spectrogram |
| freq_mask                 | Masks a random range of frequency bins                 |
| spec_augment              | Applies both time and frequency masking                |
| time_warp                 | Warps the spectrogram along the time axis              |
| add_noise                 | Adds random noise to the spectrogram                   |
| time_shift                | Shifts the spectrogram along the time axis             |
| freq_shift                | Shifts the spectrogram along the frequency axis        |
| resize_crop               | Randomly resizes and crops the spectrogram             |
| dynamic_range_compression | Applies dynamic range compression (log scaling)        |
| band_drop                 | Drops random frequency bands                           |
| patch_swap                | Swaps random patches in the spectrogram                |


## Usage

### 1. Prepare Metadata

- **Local:**  
  `metadata.txt` with lines like:

  ```
  /path/to/audio.wav|transcription text
  ```

- **HuggingFace/TSV:**  
  `mig_burmese_metadata.tsv` with columns `audio` and `transcription`.

### 2. Run Augmentation

Edit and run [`main.py`](main.py):

```python
from src.utils.pipeline import Augmentation, HF_Augmentation
from src.utils.spectogram_aug_pipeline import SpectrogramAugmentation, HF_SpectrogramAugmentation

#For waveform augmentation with local metadata:
pipeline = Augmentation("metadata.txt", "output_of_aug")
pipeline.augment(10, ["noise", "pitch"])

# For waveform augmentation with HuggingFace TSV:
pipeline = HF_Augmentation("mig_burmese_metadata.tsv", "output_of_aug")
pipeline.augment(10, ["noise", "pitch"])

# For spectrogram augmentation with local metadata:
spec_pipeline = SpectrogramAugmentation("metadata.txt", "output_of_aug")
spec_pipeline.augment(10, ["time_mask", "freq_mask"])

# For spectrogram augmentation with HuggingFace TSV:
spec_pipeline = HF_SpectrogramAugmentation("mig_burmese_metadata.tsv", "output_of_aug")
spec_pipeline.augment(10, ["time_mask", "freq_mask"])
```

- Change the percentage and augmentation methods as needed.
- Output will be in `output_of_aug/` with subfolders for each augmentation.

# 3. In Google Colab
```
!git clone https://github.com/Ye-Bhone-Lin/Audio_Augmentation.git
```

#### Data Format

```python
from Audio_Augmentation.src.dataset_preparation.data_format_to_aug import AudioTSVWriter

writer = AudioTSVWriter(
    output_dir="audio_files",
    tsv_path="mig_burmese_metadata.tsv"
)

writer.write_split(speech_data['test'])  # or ['train'] speech_data = huggingface dataset
```

#### Wav Augmentation


```python
from Audio_Augmentation.src.wave_augmentation.pipeline import  Augmentation, HF_Augmentation

tech = ["noise", "pitch","loudness","shift",'speed']
pipeline = HF_Augmentation("tsv file", "output_path")
pipeline.augment(1, tech) # 1 = augmentation percentage
```

#### Spec Augmentation

```python
from Audio_Augmentation.src.spectogram_augmentation.spec_after_wav_aug import SpecAugmentationAfterWav

base_input_dir = "output_path"  # folder containing speed/, pitch/, etc.
base_output_dir = "spec_augmented"

spec_methods = ['time_mask', 'freq_mask']

pipeline = SpecAugmentationAfterWav(base_input_dir, base_output_dir)
pipeline.augment(spec_methods)
```

---

## Output

- **Augmented Audio:**  
  Saved in `output_of_aug/<method>/<method>_<original_filename>.wav`
- **Augmented Metadata:**  
  `output_of_aug/aug_metadata.txt` with lines like:
  ```
  /abs/path/to/output_of_aug/noise/noise_file1.wav|transcription text
  /abs/path/to/output_of_aug/pitch/pitch_file1.wav|transcription text
  ```

---

## Dependencies

- `numpy`
- `soundfile`
- `librosa`
- `nlpaug`
- `random`
- `csv`
- `pathlib`

Install with:

```sh
pip install numpy soundfile librosa nlpaug
```

---

## Updated

- Add new augmentation methods in `pipeline.py` or `spectogram_aug_pipeline.py`.
- Use your own metadata or HuggingFace datasets.
- Added to apply spec augmentation after wav form augmentation (First run wav form augmentation after that use spec_after_wav_aug.py)
---
