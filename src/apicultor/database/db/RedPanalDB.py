# -*- coding: UTF-8 -*-

import json
import sys
import requests
import random
import os
import ffmpeg
import shutil
from urllib.parse import urlparse
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import logging

from mir.db.api import MirDbApi

logger = logging.getLogger(__name__)

# RedPanal API
class RedPanalDB(MirDbApi):
    """RedPanal database API client for audio retrieval and management."""
    
    def __init__(self, base_url: str = "https://redpanal.org/api/audio", 
                 local_url: str = "http://127.0.0.1:5000"):
        """Initialize RedPanal database client.
        
        Args:
            base_url: RedPanal API base URL
            local_url: Local RedPanal instance URL (fallback)
        """
        self.__api_key = ""
        self.__base_url = base_url
        self.__local_url = local_url
        self.download_dir = Path("./sounds")
        self.download_dir.mkdir(exist_ok=True)

    def set_api_key(self, api_key: str) -> None:
        """Set API key for authentication."""
        self.__api_key = api_key

    def list_audios(self, genre: Optional[str] = None, tag: Optional[str] = None, 
                   page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """List audios from RedPanal with optional filters.
        
        Args:
            genre: Filter by genre
            tag: Filter by tag
            page: Page number
            page_size: Number of items per page
            
        Returns:
            Dictionary with audio list or error
        """
        params = {
            "page": page,
            "page_size": page_size
        }
        if genre:
            params["genre"] = genre
        if tag:
            params["tag"] = tag

        url = f"{self.__base_url}/list/"
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error listing audios: {e}")
            return {"error": str(e)}

    def get_audio_detail(self, audio_id: int) -> Dict[str, Any]:
        """Get audio details by ID.
        
        Args:
            audio_id: Audio ID
            
        Returns:
            Dictionary with audio details or error
        """
        url = f"{self.__base_url}/{audio_id}/"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error getting audio detail for ID {audio_id}: {e}")
            return {"error": str(e)}

    def convert_to_wav(self, input_path: str) -> str:
        """Convert input audio file to .wav if not already in .wav format.
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Path to WAV file
            
        Raises:
            RuntimeError: If conversion fails
        """
        if input_path.lower().endswith('.wav'):
            return input_path
        
        output_path = os.path.splitext(input_path)[0] + ".wav"
        try:
            (
                ffmpeg.input(input_path)
                .output(output_path, acodec='pcm_s16le', ac=1, ar='44100')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else str(e)
            raise RuntimeError(f"ffmpeg conversion failed: {error_msg}")

    def download_sample(self, soundfile_url: str) -> Tuple[str, Dict[str, Any]]:
        """Download a sound from RedPanal.org by its URL and convert it to WAV.
        
        Args:
            soundfile_url: URL to the audio file
            
        Returns:
            Tuple of (info_text, metadata_dict)
            
        Raises:
            Exception: If download or conversion fails
        """
        parsed_url = urlparse(soundfile_url)
        filename = os.path.basename(parsed_url.path)
        local_path = self.download_dir / filename
        
        try:
            # Download the file
            with requests.get(soundfile_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            
            # Convert to WAV
            wav_path = self.convert_to_wav(str(local_path))
            
            info = {
                "original_file": filename,
                "wav_file": os.path.basename(wav_path),
                "wav_path": wav_path,
                "size_bytes": os.path.getsize(wav_path)
            }
            
            info_txt = f"Downloaded and converted {filename} to {os.path.basename(wav_path)} ({info['size_bytes']} bytes)"
            return info_txt, info
            
        except Exception as e:
            logger.error(f"Error downloading sample from {soundfile_url}: {e}")
            raise

    def search_by_mir(self, mir_state: Dict[str, Any]) -> Optional[Tuple[str, str, int]]:
        """Search for audio by MIR state parameters.
        
        Args:
            mir_state: Dictionary with MIR parameters
            
        Returns:
            Tuple of (file_path, author, sound_id) or None
        """
        try:
            # Try RedPanal API first
            audios = self.list_audios(page_size=50)
            if "error" not in audios and "results" in audios:
                results = audios["results"]
                if results:
                    # Choose a random audio
                    chosen_audio = random.choice(results)
                    audio_id = chosen_audio.get("id", 0)
                    author = chosen_audio.get("user", {}).get("username", "Unknown")
                    
                    # Get audio detail to find download URL
                    detail = self.get_audio_detail(audio_id)
                    if "error" not in detail and "audio" in detail:
                        audio_url = detail["audio"]
                        info_txt, metadata = self.download_sample(audio_url)
                        return metadata["wav_path"], author, audio_id
            
            # Fallback to local instance
            return self._search_local_samples(mir_state)
            
        except Exception as e:
            logger.error(f"Error in search_by_mir: {e}")
            return self._search_local_samples(mir_state)

    def _search_local_samples(self, mir_state: Dict[str, Any]) -> Optional[Tuple[str, str, int]]:
        """Search local samples as fallback.
        
        Args:
            mir_state: Dictionary with MIR parameters
            
        Returns:
            Tuple of (file_path, author, sound_id) or None
        """
        try:
            call = '/list/samples'  # gets only wav files because SuperCollider
            response = requests.get(self.__local_url + call, timeout=5)
            response.raise_for_status()
            
            audio_files = []
            for file in response.text.split('\n'):
                if len(file) > 0:  # avoid null paths
                    audio_files.append(file)
            
            if not audio_files:
                return None
                
            # Choose file randomly, ensuring it exists and has reasonable size
            for _ in range(len(audio_files)):
                file_chosen = audio_files[random.randint(0, len(audio_files)-1)]
                if os.path.exists(file_chosen) and os.path.getsize(file_chosen) > 1000:
                    return file_chosen, "Unknown", 0
                    
            return None
            
        except Exception as e:
            logger.error(f"Error searching local samples: {e}")
            return None

    def get_one_by_mir(self, mir_state: Dict[str, Any]) -> Optional[Tuple[str, str, int]]:
        """Get one audio file by MIR state.
        
        Args:
            mir_state: Dictionary with MIR parameters
            
        Returns:
            Tuple of (file_path, author, sound_id) or None
        """
        return self.search_by_mir(mir_state)

    def search_by_id(self, audio_id: int) -> Optional[Dict[str, Any]]:
        """Search for audio by ID.
        
        Args:
            audio_id: Audio ID
            
        Returns:
            Audio details dictionary or None
        """
        return self.get_audio_detail(audio_id)

    def search_by_content(self, content: str) -> Dict[str, Any]:
        """Search for audio by content/tag.
        
        Args:
            content: Search query
            
        Returns:
            Search results dictionary
        """
        return self.list_audios(tag=content)
