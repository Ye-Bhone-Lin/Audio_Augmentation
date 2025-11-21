"""
Dataset preparatin on a HuggingFace dataset's audio column for Audio Augmentation
Author: Ye Bhone Lin
Date: 21st Nov 2025
"""


import os
import csv
import soundfile as sf


class AudioTSVWriter:
    def __init__(self, output_dir="audio_files", tsv_path="metadata.tsv"):
        self.output_dir = output_dir
        self.tsv_path = tsv_path

        os.makedirs(self.output_dir, exist_ok=True)

    def _sanitize_text(self, text: str) -> str:
        """Removes tabs/newlines to avoid breaking TSV format."""
        return text.replace("\t", " ").replace("\n", " ")

    def write_split(self, split):
        """
        Convert a HuggingFace dataset split into:
        - Saved WAV files
        - TSV metadata file
        """

        with open(self.tsv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["audio", "prompt"])

            for i, example in enumerate(split):
                audio_info = example["audio"]
                prompt = self._sanitize_text(example["prompt"])

                original_filename = audio_info.get("path")
                if original_filename:
                    filename = os.path.basename(original_filename)
                else:
                    filename = f"audio_{i}.wav"

                filepath = os.path.join(self.output_dir, filename)

                sf.write(filepath, audio_info["array"], audio_info["sampling_rate"])

                writer.writerow([filepath, prompt])

#writer = AudioTSVWriter(
#    output_dir="audio_files",
#    tsv_path="mig_burmese_metadata.tsv"
#)

#writer.write_split(speech_data['test'])  # or ['train'] speech_data = huggingface dataset
