"""
YouTube Transcript Extraction Module

This module provides functionality to extract raw text transcripts from YouTube videos.
It uses the youtube-transcript-api library to fetch transcripts and returns clean text
without timestamps.
"""

from youtube_transcript_api import YouTubeTranscriptApi
import re


def get_youtube_transcript(url):
    """
    Extract the full transcript from a YouTube video.
    
    Args:
        url (str): The full YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)
    
    Returns:
        str: A single string containing the entire video transcript with timestamps removed
    
    Raises:
        Exception: If the video ID cannot be extracted or transcript is unavailable
    
    Example:
        >>> url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        >>> transcript = get_youtube_transcript(url)
        >>> print(transcript)
    """
    
    # Extract video ID from URL
    # Handles formats like:
    # - https://www.youtube.com/watch?v=VIDEO_ID
    # - https://youtu.be/VIDEO_ID
    # - https://www.youtube.com/watch?v=VIDEO_ID&feature=...
    
    video_id = None
    
    # Pattern 1: Standard YouTube URL
    if "watch?v=" in url:
        video_id = url.split("watch?v=")[1].split("&")[0]
    # Pattern 2: Shortened youtu.be URL
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    
    print(f"Extracting transcript for video ID: {video_id}")
    
    try:
        # Create API instance
        ytt_api = YouTubeTranscriptApi()
        
        # Fetch the transcript (returns FetchedTranscript object)
        # This will try to get English transcript by default
        fetched_transcript = ytt_api.fetch(video_id)
        
        # Convert to raw data (list of dictionaries with 'text', 'start', 'duration')
        transcript_data = fetched_transcript.to_raw_data()
        
        # Extract only the 'text' field from each entry and join into single string
        # Each entry looks like: {'text': 'Hello', 'start': 0.0, 'duration': 1.5}
        text_segments = [entry['text'] for entry in transcript_data]
        
        # Join all text segments with spaces
        full_transcript = ' '.join(text_segments)
        
        # Clean up extra whitespace
        full_transcript = ' '.join(full_transcript.split())
        
        print(f"Successfully extracted {len(text_segments)} text segments")
        print(f"Total transcript length: {len(full_transcript)} characters")
        
        return full_transcript
        
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")


if __name__ == "__main__":
    # Test with the first YouTube video ever uploaded
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    
    print("=" * 70)
    print("Testing YouTube Transcript Extraction")
    print("=" * 70)
    print(f"\nURL: {test_url}")
    print("\nFetching transcript...\n")
    
    try:
        transcript = get_youtube_transcript(test_url)
        
        print("\n" + "=" * 70)
        print("TRANSCRIPT EXTRACTED SUCCESSFULLY")
        print("=" * 70)
        print("\nFirst 500 characters:")
        print("-" * 70)
        print(transcript[:500])
        print("-" * 70)
        print("\nLast 300 characters:")
        print("-" * 70)
        print(transcript[-300:])
        print("-" * 70)
        print(f"\nTotal length: {len(transcript)} characters")
        print(f"Word count: {len(transcript.split())} words")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
