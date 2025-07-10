import nlpaug.augmenter.audio as naa
import numpy as np

class AudioAugmentationPipeline:
    def __init__(self, sr):
        self.sr = sr
        self.augmenters = {
            'loudness': naa.LoudnessAug(),
            'crop': naa.CropAug(sampling_rate=sr),
            'mask': naa.MaskAug(
                sampling_rate=sr,
                zone=(0.0, 1.0),
                coverage=0.1,
                mask_with_noise=False,
                stateless=True
            ),
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
            raise ValueError(f"Input audio too short for augmentation '{augmenter_name}'.")

        augmented_data = self.augmenters[augmenter_name].augment(data)
        return np.array(augmented_data)



