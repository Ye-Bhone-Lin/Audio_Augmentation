import os
import sys
import zipfile
import shutil
import csv
from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.pipeline import AudioAugmentationPipeline
import librosa
import soundfile as sf

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Get the absolute path of the directory where this script is located
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, 'uploads')
app.config['AUGMENTED_FOLDER'] = os.path.join(APP_ROOT, 'augmented_files')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB max upload size

# Ensure the upload and augmented folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUGMENTED_FOLDER'], exist_ok=True)

# Global variable to store processing results
processing_results = {}

def debug_zip_contents(zip_path):
    """Debug function to show zip file contents"""
    debug_info = {
        'zip_exists': os.path.exists(zip_path),
        'zip_size': os.path.getsize(zip_path) if os.path.exists(zip_path) else 0,
        'contents': []
    }
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            debug_info['contents'] = zip_ref.namelist()
    except Exception as e:
        debug_info['error'] = str(e)
    
    return debug_info

def debug_metadata_file(metadata_path):
    """Debug function to analyze metadata file"""
    debug_info = {
        'exists': os.path.exists(metadata_path),
        'size': os.path.getsize(metadata_path) if os.path.exists(metadata_path) else 0,
        'first_lines': [],
        'total_lines': 0,
        'encoding_issues': False
    }
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                debug_info['total_lines'] = len(lines)
                debug_info['first_lines'] = [line.strip() for line in lines[:5]]
        except UnicodeDecodeError:
            debug_info['encoding_issues'] = True
            try:
                with open(metadata_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
                    debug_info['total_lines'] = len(lines)
                    debug_info['first_lines'] = [line.strip() for line in lines[:5]]
                    debug_info['suggested_encoding'] = 'latin-1'
            except Exception as e:
                debug_info['error'] = str(e)
        except Exception as e:
            debug_info['error'] = str(e)
    
    return debug_info

@app.route('/debug/<session_id>')
def debug_session(session_id):
    """Debug endpoint to show processing details"""
    if session_id in processing_results:
        return jsonify(processing_results[session_id])
    return jsonify({'error': 'Session not found'}), 404

@app.route('/download/<session_id>/<format>')
def download_results(session_id, format):
    """Download results in different formats"""
    if session_id not in processing_results:
        flash('Session not found', 'error')
        return redirect(url_for('index'))
    
    results = processing_results[session_id]
    if not results.get('success'):
        flash('No successful processing results to download', 'error')
        return redirect(url_for('index'))
    
    # Create temporary file for download
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format == 'txt':
        filename = f'augmented_metadata_{timestamp}.txt'
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            for entry in results['metadata_entries']:
                f.write(f"{entry['file_path']}\t{entry['transcription']}\n")
                
    elif format == 'tsv':
        filename = f'augmented_metadata_{timestamp}.tsv'
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        
        with open(temp_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['File Path', 'Transcription', 'Augmentation Type', 'Original File'])
            for entry in results['metadata_entries']:
                writer.writerow([
                    entry['file_path'], 
                    entry['transcription'], 
                    entry['augmentation_type'],
                    entry['original_file']
                ])
                
    elif format == 'csv':
        filename = f'augmented_metadata_{timestamp}.csv'
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        
        with open(temp_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['File Path', 'Transcription', 'Augmentation Type', 'Original File'])
            for entry in results['metadata_entries']:
                writer.writerow([
                    entry['file_path'], 
                    entry['transcription'], 
                    entry['augmentation_type'],
                    entry['original_file']
                ])
    else:
        flash('Invalid format requested', 'error')
        return redirect(url_for('index'))
    
    return send_file(temp_path, as_attachment=True, download_name=filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Generate session ID for this processing session
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        # Initialize processing results
        processing_results[session_id] = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'debug_info': {},
            'success': False,
            'error_message': None,
            'processed_count': 0,
            'error_count': 0,
            'metadata_entries': [],
            'output_files': []
        }
        
        try:
            # Validate file upload
            if 'file' not in request.files:
                processing_results[session_id]['error_message'] = 'No file uploaded'
                return render_template('index.html', error='No file part', session_id=session_id)
            
            file = request.files['file']
            if file.filename == '' or not file.filename.endswith('.zip'):
                processing_results[session_id]['error_message'] = 'Invalid file type'
                return render_template('index.html', error='Please upload a .zip file', session_id=session_id)
            
            # Validate augmentations
            augmentations = request.form.getlist('augmentations')
            if not augmentations:
                processing_results[session_id]['error_message'] = 'No augmentations selected'
                return render_template('index.html', error='Please select at least one augmentation', session_id=session_id)
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(zip_path)
            
            # Debug zip file
            zip_debug = debug_zip_contents(zip_path)
            processing_results[session_id]['debug_info']['zip'] = zip_debug
            
            # Create extraction directory
            extract_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'extracted_{session_id}')
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir)
            
            # Extract zip file
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                processing_results[session_id]['debug_info']['extraction'] = 'Success'
            except Exception as e:
                processing_results[session_id]['error_message'] = f'Failed to extract zip: {str(e)}'
                return render_template('index.html', error=f'Failed to extract zip: {str(e)}', session_id=session_id)
            
            # Find metadata file (try different possible locations)
            metadata_paths = [
                os.path.join(extract_dir, 'metadata.txt'),
                os.path.join(extract_dir, 'metadata.tsv'),
                os.path.join(extract_dir, 'Wavs', 'metadata.txt'),
            ]
            
            metadata_path = None
            for path in metadata_paths:
                if os.path.exists(path):
                    metadata_path = path
                    break
            
            if not metadata_path:
                # List all files in extract directory for debugging
                all_files = []
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        all_files.append(os.path.relpath(os.path.join(root, file), extract_dir))
                
                processing_results[session_id]['debug_info']['extracted_files'] = all_files
                processing_results[session_id]['error_message'] = 'metadata.txt not found'
                return render_template('index.html', 
                                     error='metadata.txt not found in the zip file. Please check the file structure.', 
                                     session_id=session_id)
            
            # Debug metadata file
            metadata_debug = debug_metadata_file(metadata_path)
            processing_results[session_id]['debug_info']['metadata'] = metadata_debug
            
            # Create output directory
            output_base_dir = os.path.join(app.config['AUGMENTED_FOLDER'], f'output_{session_id}')
            if os.path.exists(output_base_dir):
                shutil.rmtree(output_base_dir)
            os.makedirs(output_base_dir)
            
            # Process audio files
            processed_count = 0
            error_count = 0
            metadata_entries = []
            
            # Determine audio files directory
            audio_dirs = [
                os.path.join(extract_dir, 'Wavs'),
                os.path.join(extract_dir, 'wavs'),
                os.path.join(extract_dir, 'audio'),
                extract_dir
            ]
            
            audio_base_dir = None
            for dir_path in audio_dirs:
                if os.path.exists(dir_path) and any(f.endswith('.wav') for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))):
                    audio_base_dir = dir_path
                    break
            
            processing_results[session_id]['debug_info']['audio_base_dir'] = audio_base_dir
            
            if not audio_base_dir:
                processing_results[session_id]['error_message'] = 'No audio files found'
                return render_template('index.html', 
                                     error='No .wav files found in the zip file', 
                                     session_id=session_id)
            
            # Read and process metadata
            encoding = 'utf-8'
            if metadata_debug.get('encoding_issues'):
                encoding = metadata_debug.get('suggested_encoding', 'latin-1')
            
            with open(metadata_path, 'r', encoding=encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Try different separators
                    parts = None
                    separator = None
                    for sep in ['\t', '|', ',', ';']:
                        test_parts = line.split(sep)
                        if len(test_parts) >= 2:
                            parts = test_parts
                            separator = sep
                            break
                    
                    if not parts or len(parts) < 2:
                        print(f"[WARNING] Line {line_num}: Invalid format, skipping: {line}")
                        error_count += 1
                        continue
                    
                    audio_filename = parts[0].strip()
                    transcription = parts[1].strip()
                    
                    # Find audio file
                    audio_path = os.path.join(audio_base_dir, audio_filename)
                    if not os.path.exists(audio_path):
                        print(f"[WARNING] Audio file not found: {audio_path}")
                        error_count += 1
                        continue
                    
                    try:
                        print(f"[INFO] Processing: {audio_filename}")
                        data, sr = librosa.load(audio_path)
                        pipeline = AudioAugmentationPipeline(sr)
                        
                        for aug_name in augmentations:
                            aug_output_dir = os.path.join(output_base_dir, aug_name)
                            os.makedirs(aug_output_dir, exist_ok=True)
                            
                            augmented_audio = pipeline.augment(data, aug_name)
                            
                            new_audio_filename = f"{aug_name}_{audio_filename}"
                            new_audio_path = os.path.join(aug_output_dir, new_audio_filename)
                            
                            sf.write(new_audio_path, augmented_audio, sr)
                            
                            # Add to metadata entries
                            relative_path = os.path.join(aug_name, new_audio_filename)
                            metadata_entries.append({
                                'file_path': relative_path,
                                'transcription': transcription,
                                'augmentation_type': aug_name,
                                'original_file': audio_filename
                            })
                        
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"[ERROR] Failed to process {audio_filename}: {str(e)}")
                        error_count += 1
                        continue
            
            # Update processing results
            processing_results[session_id]['processed_count'] = processed_count
            processing_results[session_id]['error_count'] = error_count
            processing_results[session_id]['metadata_entries'] = metadata_entries
            
            if processed_count == 0:
                processing_results[session_id]['error_message'] = f'No audio files were successfully processed. Processed: {processed_count}, Errors: {error_count}'
                return render_template('index.html', 
                                     error=f'No audio files were successfully processed. Check the debug info for details.', 
                                     session_id=session_id)
            
            # Write metadata files
            metadata_txt_path = os.path.join(output_base_dir, 'aug_metadata.txt')
            with open(metadata_txt_path, 'w', encoding='utf-8') as f:
                for entry in metadata_entries:
                    f.write(f"{entry['file_path']}\t{entry['transcription']}\n")
            
            # Collect all output files
            output_files = []
            for root, dirs, files in os.walk(output_base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    output_files.append(file_path)
            
            processing_results[session_id]['output_files'] = output_files
            processing_results[session_id]['success'] = True
            
            return render_template('index.html', 
                                 success=True,
                                 processed_count=processed_count,
                                 error_count=error_count,
                                 total_files=len(metadata_entries),
                                 session_id=session_id,
                                 output_files=output_files[:10])  # Show first 10 files
            
        except Exception as e:
            processing_results[session_id]['error_message'] = str(e)
            return render_template('index.html', error=f'Unexpected error: {str(e)}', session_id=session_id)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
