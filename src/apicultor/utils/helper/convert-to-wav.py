#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import ffmpeg
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Default audio extensions to convert
DEFAULT_EXTENSIONS = ['.mp3', '.ogg', '.flac', '.m4a', '.aac', '.wma']

Usage = "./convert-to-wav.py [DATA_PATH] [--extensions ext1,ext2,...]"


def convert_to_wav(input_path: str, output_path: str, 
                   sample_rate: int = 44100, channels: int = None) -> bool:
    """Convert audio file to WAV format using ffmpeg-python.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file
        sample_rate: Target sample rate (default: 44100)
        channels: Target number of channels (None = keep original)
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        input_stream = ffmpeg.input(input_path)
        
        # Prepare output parameters
        output_params = {
            'acodec': 'pcm_s16le',
            'ar': sample_rate
        }
        
        if channels:
            output_params['ac'] = channels
        
        (
            ffmpeg
            .output(input_stream, output_path, **output_params)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
        
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error converting {input_path}: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"Error converting {input_path}: {e}")
        return False


def convert_audio_file(input_path: str, remove_original: bool = False) -> Optional[str]:
    """Convert single audio file to WAV format.
    
    Args:
        input_path: Path to input audio file
        remove_original: Whether to remove original file after conversion
        
    Returns:
        Path to output WAV file if successful, None otherwise
    """
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return None
    
    # Generate output path
    wav_path = input_file.parent / f"{input_file.stem}.wav"
    
    # Skip if already WAV
    if input_file.suffix.lower() == '.wav':
        print(f"Skipping {input_file.name} (already WAV format)")
        return str(input_file)
    
    try:
        print(f"Converting {input_file.name} to WAV...")
        
        if not convert_to_wav(str(input_file), str(wav_path)):
            return None
        
        # Remove original file if requested
        if remove_original:
            input_file.unlink()
            print(f"Removed original file: {input_file.name}")
        
        print(f"Successfully converted {input_file.name} to {wav_path.name}")
        return str(wav_path)
        
    except Exception as e:
        logger.error(f"Error during conversion process: {e}")
        # Clean up partial file
        if wav_path.exists() and wav_path != input_file:
            try:
                wav_path.unlink()
            except Exception:
                pass
        return None


def convert_directory(data_path: str, extensions: List[str] = None, 
                     remove_original: bool = False) -> tuple:
    """Convert all audio files in directory to WAV format.
    
    Args:
        data_path: Path to directory containing audio files
        extensions: List of file extensions to convert (default: common audio formats)
        remove_original: Whether to remove original files after conversion
        
    Returns:
        Tuple of (converted_count, failed_count)
    """
    if extensions is None:
        extensions = DEFAULT_EXTENSIONS
    
    # Normalize extensions to lowercase with dots
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                 for ext in extensions]
    
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise IOError(f"Directory does not exist: {data_path}")
    
    converted_count = 0
    failed_count = 0
    
    # Walk through directory and convert files
    for subdir, dirs, files in os.walk(data_dir):
        subdir_path = Path(subdir)
        
        for filename in files:
            file_path = subdir_path / filename
            
            # Check if file has target extension
            if file_path.suffix.lower() in extensions:
                print(f"Processing {filename}")
                
                result = convert_audio_file(
                    str(file_path), 
                    remove_original=remove_original
                )
                
                if result:
                    converted_count += 1
                else:
                    failed_count += 1
                    print(f"Failed to convert {filename}")
    
    return converted_count, failed_count


def main():
    """Main conversion script."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Parse command line arguments
    data_path = "."  # Default to current directory
    extensions = DEFAULT_EXTENSIONS
    remove_original = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print(f"""
{Usage}

Convert audio files to WAV format using ffmpeg-python.

Arguments:
    DATA_PATH           Directory to process (default: current directory)
    --extensions LIST   Comma-separated list of extensions (default: mp3,ogg,flac,m4a,aac,wma)
    --remove-original   Remove original files after conversion
    --help, -h          Show this help message

Examples:
    python convert-to-wav.py
    python convert-to-wav.py /path/to/audio
    python convert-to-wav.py . --extensions mp3,ogg --remove-original
""")
            sys.exit(0)
        
        data_path = sys.argv[1]
        
        # Parse additional arguments
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == '--extensions' and i + 1 < len(sys.argv):
                extensions = [ext.strip() for ext in sys.argv[i + 1].split(',')]
                i += 2
            elif sys.argv[i] == '--remove-original':
                remove_original = True
                i += 1
            else:
                print(f"Unknown argument: {sys.argv[i]}")
                sys.exit(1)
    
    try:
        print(f"Converting audio files in: {data_path}")
        print(f"Target extensions: {', '.join(extensions)}")
        print(f"Remove originals: {remove_original}")
        print("-" * 50)
        
        converted_count, failed_count = convert_directory(
            data_path, 
            extensions, 
            remove_original
        )
        
        print("-" * 50)
        print(f"Conversion complete!")
        print(f"Successfully converted: {converted_count} files")
        print(f"Failed conversions: {failed_count} files")
        
        if failed_count > 0:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
