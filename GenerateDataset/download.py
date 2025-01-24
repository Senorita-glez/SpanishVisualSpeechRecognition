import yt_dlp
import re
import os

def sanitize_filename(filename):
    """Replace special characters in the filename with underscores."""
    sanitized = re.sub(r'[^\w\s]', '_', filename)  # Replace non-alphanumeric characters
    sanitized = sanitized.replace(' ', '_')       # Replace spaces with underscores
    return sanitized

def DownloadVideo(url, platform, save_path):
    if isinstance(url, str):
        result = {"file_path": None, "title": None, "duration": None}
        if platform == 'yt':  # YouTube platform
            try:
                # Extract video information without downloading first to sanitize title
                ydl_opts_info = {
                    'quiet': True,
                    'skip_download': True,  # Only fetch metadata
                }
                with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                    info = ydl.extract_info(url, download=False)
                    original_title = info['title']
                    sanitized_title = sanitize_filename(original_title)

                # Use sanitized title in download options
                ydl_opts = {
                    'outtmpl': f'{save_path}/{sanitized_title}.mp4',  # Sanitized title
                    'format': 'mp4',
                    'quiet': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                sanitized_file_path = f"{save_path}/{sanitized_title}.mp4"
                result["file_path"] = sanitized_file_path
                result["title"] = sanitized_title
                result["duration"] = info.get('duration')
                return result

            except Exception as e:
                print(f"An error occurred with yt-dlp: {e}")
                print("Falling back to pytubefix...")

                # Fallback to pytubefix
                try:
                    from pytubefix import YouTube
                    from pytubefix.cli import on_progress
                    yt = YouTube(url, on_progress_callback=on_progress)
                    video_stream = yt.streams.get_highest_resolution()
                    original_title = yt.title
                    sanitized_title = sanitize_filename(original_title)
                    
                    file_path = video_stream.download(output_path=save_path)
                    
                    # Rename the file with sanitized title
                    sanitized_file_path = f"{save_path}/{sanitized_title}.mp4"
                    if os.path.exists(file_path):
                        os.rename(file_path, sanitized_file_path)

                    result["file_path"] = sanitized_file_path
                    result["title"] = sanitized_title
                    result["duration"] = yt.length
                    return result
                except Exception as fallback_error:
                    print(f"An error occurred with pytubefix: {fallback_error}")
                    return None
        else:  # TikTok or other platforms
            try:
                # Similar logic as YouTube
                ydl_opts_info = {
                    'quiet': True,
                    'skip_download': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                    info = ydl.extract_info(url, download=False)
                    original_title = info['title']
                    sanitized_title = sanitize_filename(original_title)

                ydl_opts = {
                    'outtmpl': f'{save_path}/{sanitized_title}.mp4',
                    'format': 'mp4',
                    'quiet': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                sanitized_file_path = f"{save_path}/{sanitized_title}.mp4"
                result["file_path"] = sanitized_file_path
                result["title"] = sanitized_title
                result["duration"] = info.get('duration')
                return result
            except Exception as e:
                print(f"An error occurred with TikTok video: {e}")
                return None
    else:
        print("URL is not valid, must be a string.")
        return None
