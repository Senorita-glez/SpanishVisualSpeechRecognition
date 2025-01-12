import os
from download import DownloadVideo

def processVideo(url, platform, number):
    # Define save path
    savepath = f"../Data/video{number}/"
    
    # Create the directory if it doesn't exist
    os.makedirs(savepath, exist_ok=True)
    
    # Call the DownloadVideo function and handle the result
    try:
        result = DownloadVideo(url, platform, savepath)
        print(f"Download result: {result}")
    except Exception as e:
        print(f"An error occurred: {e}")


processVideo('https://youtu.be/rWlY7JiMXHs?si=xwVo0_WPVoZ92hsD', 'yt', 1)