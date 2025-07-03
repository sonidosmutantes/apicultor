#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import ffmpeg
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

Usage = "./convert_to_ogg.py [DATA_PATH]"


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert audio file to WAV format using ffmpeg-python.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, acodec='pcm_s16le', ar=44100)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error converting {input_path} to WAV: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"Error converting {input_path} to WAV: {e}")
        return False


def convert_to_ogg(input_path: str, output_path: str, quality: int = 5) -> bool:
    """Convert audio file to OGG format using ffmpeg-python.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output OGG file
        quality: OGG quality level (0-10, higher is better)
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, acodec='libvorbis', audio_bitrate='192k', ar=44100)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error converting {input_path} to OGG: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"Error converting {input_path} to OGG: {e}")
        return False


def convert_audio_file(input_path: str, remove_original: bool = True, 
                      remove_intermediate: bool = True) -> Optional[str]:
    """Convert audio file to OGG format via WAV intermediate.
    
    Args:
        input_path: Path to input audio file
        remove_original: Whether to remove original file after conversion
        remove_intermediate: Whether to remove intermediate WAV file
        
    Returns:
        Path to output OGG file if successful, None otherwise
    """
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return None
    
    # Generate output paths
    base_name = input_file.stem
    wav_path = input_file.parent / f"{base_name}.wav"
    ogg_path = input_file.parent / f"{base_name}.ogg"
    
    try:
        # Step 1: Convert to WAV
        print(f"Converting {input_file.name} to WAV...")
        if not convert_to_wav(str(input_file), str(wav_path)):
            return None
        
        # Step 2: Convert WAV to OGG
        print(f"Converting {wav_path.name} to OGG...")
        if not convert_to_ogg(str(wav_path), str(ogg_path)):
            # Clean up WAV file if OGG conversion failed
            if wav_path.exists():
                wav_path.unlink()
            return None
        
        # Step 3: Clean up files
        if remove_intermediate and wav_path.exists():
            wav_path.unlink()
            
        if remove_original and input_file != ogg_path:
            input_file.unlink()
            
        print(f"Successfully converted {input_file.name} to {ogg_path.name}")
        return str(ogg_path)
        
    except Exception as e:
        logger.error(f"Error during conversion process: {e}")
        # Clean up any partial files
        for temp_file in [wav_path, ogg_path]:
            if temp_file.exists() and temp_file != input_file:
                try:
                    temp_file.unlink()
                except Exception:
                    pass
        return None


def main():
    """Main conversion script."""
    if len(sys.argv) < 2:
        print(f"\nBad amount of input arguments\n{Usage}\n")
        sys.exit(1)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        data_path = Path(sys.argv[1])
        if not data_path.exists():
            raise IOError("Input directory must exist")
        
        # Create duration directory
        duration_dir = data_path / 'duration'
        duration_dir.mkdir(exist_ok=True)
        
        # Supported audio formats to convert
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.wma'}
        
        converted_count = 0
        failed_count = 0
        
        # Walk through directory and convert files
        for subdir in data_path.rglob('*'):
            if subdir.is_dir():
                continue
                
            if subdir.suffix.lower() in audio_extensions:
                print(f"Processing {subdir.name}")
                
                result = convert_audio_file(
                    str(subdir), 
                    remove_original=True,
                    remove_intermediate=True
                )
                
                if result:
                    converted_count += 1
                else:
                    failed_count += 1
                    print(f"Failed to convert {subdir.name}")
        
        print(f"\nConversion complete!")
        print(f"Successfully converted: {converted_count} files")
        print(f"Failed conversions: {failed_count} files")
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
