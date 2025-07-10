from src.utils.pipeline import  Augmentation

pipeline = Augmentation("metadata.txt", "output_of_aug")
pipeline.augment(1, ["noise", "pitch"])
