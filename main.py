from src.utils.pipeline import AudioAugmentationPipeline
import os 
import librosa 

dir = "D:\\ASR_Augmentation\\src\\test_data"

file_path = os.path.join(dir, "recording_20250622_180044.wav")
#print(file_path)

data, sr = librosa.load(file_path)

pipeline = AudioAugmentationPipeline(sr)

chosen_augmentation = 'pitch'  

augmented_audio = pipeline.augment(data, chosen_augmentation)

#print(augmented_audio)

