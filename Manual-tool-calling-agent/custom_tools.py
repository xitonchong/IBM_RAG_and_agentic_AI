from langchain_core.tools import tool 
import re 
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict
from pytube import Search 
import yt_dlp 
import logging 
from langchain.tools import tool

from typing import List, Dict

# suppress yt-dlp warnings 
yt_dpl_logger = logging.getLogger("yt_dlp")

@tool 
def extract_video_id(url: str) -> str: 
    '''
    Extracts the 11-character YouTube video ID from a URL. 

    args: 
        url (str): A YouTube URL containing a video ID. 
    
    Returns: 
        str: Extracted video ID or error message if parsing fails.
    '''
    # regex pattern to match video IDs
    pattern = r'(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"


@tool 
def fetch_transcript(video_id: str, language: str='en') -> str: 
    """
    Fetches the transcript of a YouTube video.
    
    Args:
        video_id (str): The YouTube video ID (e.g., "dQw4w9WgXcQ").
        language (str): Language code for the transcript (e.g., "en", "es").
    
    Returns:
        str: The transcript text or an error message.
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=[language])
        return " ".join([snippet.text for snippet in transcript.snippets])
    except Exception as e:
        return f"Error: {str(e)}"


@tool 
def search_youtube(query: str) -> List[Dict[str, str]]:
    """
    Search YouTube for videos matching the query.
    
    Args:
        query (str): The search term to look for on YouTube
        
    Returns:
        List of dictionaries containing video titles and IDs in format:
        [{'title': 'Video Title', 'video_id': 'abc123'}, ...]
        Returns error message if search fails
    """
    try:
        s = Search(query)
        return [
            {
                "title": yt.title,
                "video_id": yt.video_id,
                "url": f"https://youtu.be/{yt.video_id}"
            }
            for yt in s.results
        ]
    except Exception as e:
        return f"Error: {str(e)}"


# defining metadata extracting tool 
@tool
def get_full_metadata(url: str) -> dict:
    """Extract metadata given a YouTube URL, including title, views, duration, channel, likes, comments, and chapters."""
    with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'title': info.get('title'),
            'views': info.get('view_count'),
            'duration': info.get('duration'),
            'channel': info.get('uploader'),
            'likes': info.get('like_count'),
            'comments': info.get('comment_count'),
            'chapters': info.get('chapters', [])
        }



@tool
def get_trending_videos(region_code: str) -> List[Dict]:
    """
    Fetches currently trending YouTube videos for a specific region.
    
    Args:
        region_code (str): 2-letter country code (e.g., "US", "IN", "GB")
        
    Returns:
        List of dictionaries with video details: title, video_id, channel, view_count, duration
    """
    ydl_opts = {
        'geo_bypass_country': region_code.upper(),
        'extract_flat': True,
        'quiet': True,
        'force_generic_extractor': True,
        'logger': yt_dpl_logger
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                'https://www.youtube.com/feed/trending',
                download=False
            )
            
            trending_videos = []
            for entry in info['entries']:
                video_data = {
                    'title': entry.get('title', 'N/A'),
                    'video_id': entry.get('id', 'N/A'),
                    'url': entry.get('url', 'N/A'),
                    'channel': entry.get('uploader', 'N/A'),
                    'duration': entry.get('duration', 0),
                    'view_count': entry.get('view_count', 0)
                }
                trending_videos.append(video_data)
                
            return trending_videos[:25]  # Return top 25 trending videos
            
    except Exception as e:
        return [{'error': f"Failed to fetch trending videos: {str(e)}"}]


@tool
def get_thumbnails(url: str) -> List[Dict]:
    """
    Get available thumbnails for a YouTube video using its URL.
    
    Args:
        url (str): YouTube video URL (any format)
        
    Returns:
        List of dictionaries with thumbnail URLs and resolutions in YouTube's native order
    """
    
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
            info = ydl.extract_info(url, download=False)
            
            thumbnails = []
            for t in info.get('thumbnails', []):
                if 'url' in t:
                    thumbnails.append({
                        "url": t['url'],
                        "width": t.get('width'),
                        "height": t.get('height'),
                        "resolution": f"{t.get('width', '')}x{t.get('height', '')}".strip('x')
                    })
            
            return thumbnails

    except Exception as e:
        return [{"error": f"Failed to get thumbnails: {str(e)}"}]



if __name__ == '__main__': 
    print(extract_video_id.name)
    print("----------------")
    print(extract_video_id.description)
    print("-"*10)
    print(extract_video_id.func)

    video_id = extract_video_id.run("https://www.youtube.com/watch?v=hfIUstzHs9A")
    print('video id: ', video_id)


    fetch_transcript.run("hfIUstzHs9A")

    search_out = search_youtube.run("Generative AI")

    print(search_out)
    meta_data=get_full_metadata.run("https://youtu.be/qWHaMrR5WHQ")
    print(meta_data)

    thumbnails=get_thumbnails.run("https://www.youtube.com/watch?v=qWHaMrR5WHQ")

    print(thumbnails)