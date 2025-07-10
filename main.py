from src.utils.pipeline import  Augmentation,HF_Augmentation

pipeline = Augmentation("metadata.txt", "output_of_aug")
pipeline.augment(1, ["noise", "pitch"])

pipeline = HF_Augmentation("/content/mig_burmese_metadata.tsv", "output_of_aug")
pipeline.augment(1, ["noise", "pitch"])