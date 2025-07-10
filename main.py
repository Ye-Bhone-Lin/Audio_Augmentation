from src.utils.pipeline import AudioAugmentationPipeline
import os
import librosa
import soundfile as sf

def main():
    input_dir = "src/test_data"
    output_dir = "src/augmented_data"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List of augmentations to apply
    augmentations_to_apply = ['pitch', 'speed', 'noise']

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_dir, filename)
            data, sr = librosa.load(file_path)

            pipeline = AudioAugmentationPipeline(sr)

            for aug_name in augmentations_to_apply:
                print(f"Applying {aug_name} to {filename}...")
                augmented_audio = pipeline.augment(data, aug_name)
                
                # Construct the output filename
                base_name, ext = os.path.splitext(filename)
                output_filename = f"{base_name}_{aug_name}{ext}"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save the augmented audio
                sf.write(output_path, augmented_audio, sr)
                print(f"Saved augmented file: {output_path}")

if __name__ == "__main__":
    main()


