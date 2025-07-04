# -*- coding: UTF-8 -*-

"""
Local file database provider without Flask dependencies.
Simplified version of JsonMirFilesData for core functionality.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class LocalFilesData:
    """Local file database provider for MIR data and audio files."""
    
    def __init__(self, data_path: str = "./data", samples_path: str = "./samples"):
        """Initialize local file provider.
        
        Args:
            data_path: Path to JSON metadata files
            samples_path: Path to audio sample files
        """
        self.data_path = Path(data_path)
        self.samples_path = Path(samples_path)
        self.ext_filter = ['.mp3', '.ogg', '.wav']
        
        # Create directories if they don't exist
        self.data_path.mkdir(exist_ok=True)
        self.samples_path.mkdir(exist_ok=True)
        
        logger.info(f"Initialized LocalFilesData with data_path={data_path}, samples_path={samples_path}")
    
    def get_url_audio(self, audio_id: str) -> Optional[str]:
        """Get audio file path by ID.
        
        Args:
            audio_id: Audio file identifier
            
        Returns:
            Absolute path to audio file or None if not found
        """
        for audio_file in self.data_path.rglob("*"):
            if audio_file.is_file():
                stem = audio_file.stem
                suffix = audio_file.suffix
                if suffix in self.ext_filter and stem == str(audio_id):
                    return str(audio_file.absolute())
        return None
    
    def get_list_of_files(self, files_path: Optional[str] = None) -> List[str]:
        """Get list of audio files.
        
        Args:
            files_path: Optional specific path to search (defaults to samples_path)
            
        Returns:
            List of audio file paths
        """
        search_path = Path(files_path) if files_path else self.samples_path
        audio_files = []
        
        for audio_file in search_path.rglob("*"):
            if audio_file.is_file() and audio_file.suffix in self.ext_filter:
                audio_files.append(str(audio_file))
        
        return audio_files
    
    def get_list_of_sample_files_same_cluster(self, sample_name: str) -> List[str]:
        """Get list of files in the same cluster as the given sample.
        
        Args:
            sample_name: Reference sample name
            
        Returns:
            List of similar sample paths
        """
        # TODO: Implement clustering logic
        # For now, just return the sample itself
        return [sample_name]
    
    def get_list_of_files_comparing(
        self, 
        files_path: str, 
        query_descriptor: str, 
        fixed_float_value: float, 
        comp: str = ">"
    ) -> List[str]:
        """Get list of files matching descriptor comparison.
        
        Args:
            files_path: Path to search for files
            query_descriptor: MIR descriptor to compare
            fixed_float_value: Comparison value
            comp: Comparison operator (">" or "<")
            
        Returns:
            List of matching audio file paths
        """
        comp_value = float(fixed_float_value) / 1000.0  # Convert from fixed point
        
        # Map simplified descriptor names to full JSON paths
        descriptor_mapping = {
            "HFC": "lowlevel.hfc.mean",
            "duration": "metadata.duration.mean"
        }
        
        full_descriptor = descriptor_mapping.get(query_descriptor, query_descriptor)
        matching_files = []
        search_path = Path(files_path)
        
        for json_file in search_path.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Navigate nested dictionary structure
                value = self._get_nested_value(data, full_descriptor)
                if value is not None:
                    value = float(value)
                    
                    if comp == ">" and value > comp_value:
                        wav_file = json_file.with_suffix(".wav")
                        if wav_file.exists():
                            matching_files.append(str(wav_file))
                            logger.debug(f"Found match: {json_file.stem} = {value}")
                    elif comp == "<" and value < comp_value:
                        wav_file = json_file.with_suffix(".wav")
                        if wav_file.exists():
                            matching_files.append(str(wav_file))
                            logger.debug(f"Found match: {json_file.stem} = {value}")
                            
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Error processing {json_file}: {e}")
                continue
        
        return matching_files
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Optional[Any]:
        """Get value from nested dictionary using dot notation.
        
        Args:
            data: Dictionary to search
            key_path: Dot-separated key path (e.g., "lowlevel.hfc.mean")
            
        Returns:
            Value at the key path or None if not found
        """
        keys = key_path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def is_available(self) -> bool:
        """Check if the local file provider is available.
        
        Returns:
            True if data and samples directories exist
        """
        return self.data_path.exists() and self.samples_path.exists()
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information.
        
        Returns:
            Dictionary with provider details
        """
        audio_files = self.get_list_of_files()
        json_files = list(self.data_path.rglob("*.json"))
        
        return {
            "provider": "LocalFilesData",
            "data_path": str(self.data_path),
            "samples_path": str(self.samples_path),
            "audio_files_count": len(audio_files),
            "json_files_count": len(json_files),
            "supported_formats": self.ext_filter,
            "available": self.is_available()
        }