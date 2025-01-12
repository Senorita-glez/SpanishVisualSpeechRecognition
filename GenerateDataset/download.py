import yt_dlp

def DownloadVideo(url, platform, save_path="."):
    if isinstance(url, str):
        result = {"file_path": None, "title": None, "duration": None}
        if platform == 'yt':  # YouTube platform
            try:
                # Download using yt-dlp
                ydl_opts = {
                    'outtmpl': f'{save_path}/%(title)s.%(ext)s',
                    'format': 'mp4',
                    'quiet': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    result["file_path"] = f"{save_path}/{info['title']}.{info['ext']}"
                    result["title"] = info['title']
                    result["duration"] = info['duration']
                return result
            except Exception as e:
                print(f"An error occurred with yt-dlp: {e}")
                try:
                    # Fallback to pytubefix
                    from pytubefix import YouTube
                    from pytubefix.cli import on_progress
                    yt = YouTube(url, on_progress_callback=on_progress)
                    video_stream = yt.streams.get_highest_resolution()
                    file_path = video_stream.download(output_path=save_path)
                    result["file_path"] = file_path
                    result["title"] = yt.title
                    result["duration"] = yt.length
                    return result
                except Exception as e:
                    print(f"An error occurred with pytubefix: {e}")
                    return None
        else:  # TikTok or other platforms
            try:
                ydl_opts = {
                    'outtmpl': f'{save_path}/%(title)s.%(ext)s',
                    'format': 'mp4',
                    'quiet': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    result["file_path"] = f"{save_path}/{info['title']}.{info['ext']}"
                    result["title"] = info['title']
                    result["duration"] = info['duration']
                return result
            except Exception as e:
                print(f"An error occurred with TikTok video: {e}")
                return None
    else:
        print("URL is not valid, must be a string.")
        return None
