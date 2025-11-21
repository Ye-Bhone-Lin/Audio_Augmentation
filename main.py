from src.wave_augmentation.pipeline import  Augmentation,HF_Augmentation
from src.spectogram_augmentation.spectogram_aug_pipeline import  SpectrogramAugmentation,HF_SpectrogramAugmentation

#pipeline = Augmentation("metadata.txt", "output_of_aug")
#pipeline.augment(1, ["noise", "pitch"])

#pipeline = HF_Augmentation("/content/mig_burmese_metadata.tsv", "output_of_aug")
#pipeline.augment(1, ["noise", "pitch"])

#spec_pipeline = SpectrogramAugmentation("metadata.txt", "output_of_aug")
#spec_pipeline.augment(1, ["time_mask", "freq_mask"])

spec_pipeline = HF_SpectrogramAugmentation("mig_burmese_metadata.tsv", "output_of_aug")
spec_pipeline.augment(1, ["time_mask", "freq_mask"])
