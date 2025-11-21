from src.wave_augmentation.pipeline import Augmentation, HF_Augmentation
from src.spectogram_augmentation.spectogram_aug_pipeline import SpectrogramAugmentation, HF_SpectrogramAugmentation

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

