# 🎵 Audio Augmentation Web App

A powerful Flask web application for performing audio augmentation on uploaded audio files with their corresponding transcriptions.

## ✨ Features

- **Modern UI**: Beautiful, responsive interface with gradient backgrounds and smooth animations
- **Multiple Augmentations**: 8 different audio augmentation techniques
- **Flexible Input**: Supports various metadata formats (tab, pipe, comma, semicolon separated)
- **Multiple Export Formats**: Download results as TXT, TSV, or CSV
- **Debug Mode**: Built-in debugging tools to troubleshoot issues
- **Progress Tracking**: Real-time processing feedback
- **Batch Processing**: Handle large datasets efficiently

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd "/Users/htutkoko/Study Data/LU Lab/ASR_Augmentation"
pipenv install
```

### 2. Run the Application
```bash
pipenv run python web_app/app.py
```

### 3. Open Your Browser
Navigate to `http://localhost:5000`

## 📁 Required File Structure

Your zip file should contain:

```
your_data.zip
├── metadata.txt          # Metadata file (see format below)
└── Wavs/                 # Directory containing all .wav files
    ├── recording1.wav
    ├── recording2.wav
    └── ...
```

### Metadata Format

The app supports multiple separator formats:

**Tab-separated (recommended):**
```
recording_20250624T130323Z.wav	မင်္ဂလာပါ ။
recording_20250624T130336Z.wav	ခင်ဗျား က မစ္စတာ ထွန်းထွန်း ဖြစ် ပါ သလား ။
```

**Pipe-separated:**
```
recording1.wav|Hello world
recording2.wav|How are you
```

**Comma-separated:**
```
recording1.wav,Hello world
recording2.wav,How are you
```

## 🎛️ Available Augmentations

| Augmentation | Description | Icon |
|--------------|-------------|------|
| **Pitch** | Modify audio pitch | 🎵 |
| **Speed** | Change playback speed | ⚡ |
| **Noise** | Add background noise | 🔊 |
| **Loudness** | Adjust volume level | 🔉 |
| **Crop** | Crop audio segments | ✂️ |
| **Mask** | Mask audio portions | 👁️‍🗨️ |
| **Shift** | Time-shift audio | ↔️ |
| **VTLP** | Vocal Tract Length Perturbation | 〰️ |

## 📥 Download Formats

After successful processing, you can download results in three formats:

### TXT Format
Simple tab-separated format:
```
pitch/pitch_recording1.wav	Hello world
noise/noise_recording1.wav	Hello world
```

### TSV Format
Tab-separated with headers:
```
File Path	Transcription	Augmentation Type	Original File
pitch/pitch_recording1.wav	Hello world	pitch	recording1.wav
```

### CSV Format
Comma-separated with headers:
```
File Path,Transcription,Augmentation Type,Original File
pitch/pitch_recording1.wav,Hello world,pitch,recording1.wav
```

## 🐛 Troubleshooting

### Issue: "No audio files were successfully processed. Processed: 0, Errors: 0"

This usually indicates a file structure or format issue. Here's how to debug:

1. **Check File Structure**: Ensure your zip contains:
   - `metadata.txt` in the root
   - `Wavs/` folder with audio files

2. **Use Debug Mode**: If you get an error, click the "Show Debug Info" button to see detailed information about:
   - Zip file contents
   - Metadata file analysis
   - Audio file locations
   - Processing errors

3. **Common Issues**:
   - **Wrong audio folder name**: Should be `Wavs` (capital W)
   - **Metadata format**: Ensure proper separation (tab, pipe, comma, or semicolon)
   - **File name mismatch**: Audio filenames in metadata must match actual files
   - **Encoding issues**: Try saving metadata as UTF-8

### Issue: "metadata.txt not found"

The app looks for metadata files in these locations:
1. `metadata.txt` (root of zip)
2. `metadata.tsv` (root of zip)
3. `Wavs/metadata.txt` (inside Wavs folder)

### Issue: "No .wav files found"

The app searches for audio files in these directories:
1. `Wavs/` (recommended)
2. `wavs/` (lowercase)
3. `audio/`
4. Root of zip file

### Issue: Processing fails for specific files

Common causes:
- **Corrupted audio files**: Check if files play correctly
- **Unsupported format**: Ensure files are .wav format
- **File permissions**: Check file access permissions
- **Memory issues**: Very large files might cause memory problems

## 🔧 Advanced Usage

### Custom Session Management

Each processing session gets a unique ID. You can access debug information using:
```
http://localhost:5000/debug/SESSION_ID
```

### API Endpoints

- `GET /`: Main interface
- `POST /`: Process uploaded files
- `GET /debug/<session_id>`: Debug information
- `GET /download/<session_id>/<format>`: Download results

## 📊 Output Structure

After processing, files are organized as:

```
output_SESSION_ID/
├── aug_metadata.txt                    # Main metadata file
├── pitch/                             # Pitch augmentation
│   ├── pitch_recording1.wav
│   └── pitch_recording2.wav
├── noise/                             # Noise augmentation
│   ├── noise_recording1.wav
│   └── noise_recording2.wav
└── mask/                              # Mask augmentation
    ├── mask_recording1.wav
    └── mask_recording2.wav
```

## 🎨 UI Features

- **Drag & Drop**: Easy file upload
- **Visual Feedback**: Loading animations and progress indicators
- **Responsive Design**: Works on desktop and mobile
- **Dark/Light Theme**: Automatic theme adaptation
- **Interactive Cards**: Click to select augmentations
- **Real-time Validation**: Immediate feedback on form inputs

## 🔒 Security Notes

- Files are processed in isolated session directories
- Temporary files are automatically cleaned up
- No persistent storage of user data
- Session-based processing prevents conflicts

## 🚀 Performance Tips

- **Batch Size**: Process files in smaller batches for better performance
- **Memory**: Ensure sufficient RAM for large audio files
- **Storage**: Ensure adequate disk space for output files
- **Network**: Use stable internet connection for uploads

## 📝 Changelog

### Version 2.0 (Current)
- ✅ Enhanced UI with modern design
- ✅ Multiple export formats (TXT, TSV, CSV)
- ✅ Advanced debugging tools
- ✅ Flexible metadata format support
- ✅ Session-based processing
- ✅ Improved error handling
- ✅ Real-time progress feedback

### Version 1.0
- Basic Flask app with simple UI
- Limited augmentation options
- Basic error handling

## 🤝 Support

If you encounter issues:

1. **Check the Debug Info**: Use the debug button for detailed error information
2. **Verify File Structure**: Ensure your zip file follows the required structure
3. **Test with Sample Data**: Try with a small sample first
4. **Check Console Logs**: Look at browser console for JavaScript errors

## 📄 License

This project is for educational and research purposes.

---

**Happy Augmenting! 🎵✨**
