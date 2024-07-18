import os
import subprocess
from dotenv import load_dotenv

load_dotenv('/var/www/html/visionvortex.com.br/.env')
rtsp_url = os.getenv('RTSP')
output_dir = '/var/www/html/visionvortex.com.br/hls'
os.makedirs(output_dir, exist_ok=True)

ffmpeg_command = [
    'ffmpeg',
    '-i', rtsp_url,
    '-c:v', 'libx264',
    '-hls_time', '10',
    '-hls_list_size', '5',
    '-hls_flags', 'delete_segments',
    f'{output_dir}/stream.m3u8'
]

subprocess.run(ffmpeg_command)