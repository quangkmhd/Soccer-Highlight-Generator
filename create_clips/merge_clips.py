import os
import re
import subprocess
import shutil
import argparse
from datetime import datetime, time

def time_to_seconds(t_str):
    """Converts time string HH-MM-SS to seconds."""
    parts = list(map(int, t_str.split('-')))
    return parts[0] * 3600 + parts[1] * 60 + parts[2]

def parse_filename(filename):
    """Parses filename to get start and end times in seconds."""
    match = re.match(r'(\d{1,2}-\d{1,2}-\d{1,2})--(\d{1,2}-\d{1,2}-\d{1,2})', os.path.basename(filename))
    if not match:
        return None, None
    
    start_str, end_str = match.groups()
    start_sec = time_to_seconds(start_str)
    end_sec = time_to_seconds(end_str)
    
    return start_sec, end_sec

def merge_videos(input_dir, output_file, temp_dir='temp_clips_for_merge'):
    """
    Finds, sorts, de-overlaps, and merges video clips into a single file.
    """
    print(f"Bắt đầu quá trình gộp video từ thư mục: {input_dir}")

    # Check for ffmpeg
    if not shutil.which("ffmpeg"):
        print("LỖI: ffmpeg không được tìm thấy. Vui lòng cài đặt ffmpeg và đảm bảo nó nằm trong PATH của hệ thống.")
        return

    # 1. Find all video files recursively
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_files.append(os.path.join(root, file))

    if not video_files:
        print(f"Không tìm thấy file .mp4 nào trong thư mục: {input_dir}")
        return

    print(f"Tìm thấy {len(video_files)} video clip.")

    # 2. Parse filenames and sort clips by start time
    clips = []
    for f_path in video_files:
        start_sec, end_sec = parse_filename(f_path)
        if start_sec is not None:
            clips.append({'path': f_path, 'start': start_sec, 'end': end_sec})
    
    clips.sort(key=lambda x: x['start'])

    # 3. Create a temporary directory for processed clips
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    print(f"Đã tạo thư mục tạm: {temp_dir}")

    # 4. Process and trim clips to avoid overlap
    processed_clip_paths = []
    last_timeline_end_sec = 0
    clip_counter = 0

    for clip in clips:
        clip_start_in_source = clip['start']
        clip_end_in_source = clip['end']

        # Determine the portion of the clip to use
        trim_start_sec = max(clip_start_in_source, last_timeline_end_sec)
        
        if trim_start_sec < clip_end_in_source:
            # This clip has new content to add
            duration_to_trim = clip_end_in_source - trim_start_sec
            # The start point within the clip file itself
            offset_in_clip_file = trim_start_sec - clip_start_in_source

            clip_counter += 1
            output_clip_path = os.path.join(temp_dir, f'clip_{clip_counter:04d}.mp4')
            
            print(f"Đang xử lý clip {clip_counter}: {os.path.basename(clip['path'])} | Lấy đoạn từ {trim_start_sec}s đến {clip_end_in_source}s")

            # Use ffmpeg to trim the clip
            cmd = [
                'ffmpeg',
                '-ss', str(offset_in_clip_file),
                '-i', clip['path'],
                '-t', str(duration_to_trim),
                '-c:v', 'libx264', # Re-encode for compatibility
                '-preset', 'veryfast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                output_clip_path
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                processed_clip_paths.append(output_clip_path)
                last_timeline_end_sec = clip_end_in_source
            except subprocess.CalledProcessError as e:
                print(f"Lỗi khi xử lý clip {clip['path']}:")
                print(e.stderr)
                continue

    if not processed_clip_paths:
        print("Không có clip nào để gộp sau khi xử lý trùng lặp.")
        shutil.rmtree(temp_dir)
        return

    print(f"\nĐã xử lý xong {len(processed_clip_paths)} clip. Bắt đầu gộp thành video cuối cùng...")

    # 5. Create a file list for ffmpeg concatenation
    concat_list_path = os.path.join(temp_dir, 'concat_list.txt')
    with open(concat_list_path, 'w') as f:
        for p in processed_clip_paths:
            # The path in the list file must be relative to the list file's directory.
            clip_filename = os.path.basename(p)
            # Use forward slashes for compatibility.
            safe_path = clip_filename.replace('\\', '/')
            f.write(f"file '{safe_path}'\n")

    # 6. Run ffmpeg to concatenate all processed clips
    concat_cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list_path,
        '-c', 'copy', # Can copy streams as they were re-encoded identically
        output_file
    ]

    try:
        # Overwrite output file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
        subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
        print(f"\nTHÀNH CÔNG! Video đã được gộp và lưu tại: {output_file}")
    except subprocess.CalledProcessError as e:
        print("\nLỗi trong quá trình gộp video cuối cùng:")
        print(e.stderr)

    # 7. Clean up the temporary directory
    finally:
        shutil.rmtree(temp_dir)
        print(f"Đã xóa thư mục tạm: {temp_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gộp các video clip thành một video hoàn chỉnh, xử lý các đoạn bị trùng lặp.')
    parser.add_argument(
        'input_dir', 
        help='Thư mục chứa các video clip cần gộp (sẽ tìm kiếm trong các thư mục con).'
    )
    parser.add_argument(
        '-o', '--output',
        default=f'merged_video_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4',
        help='Tên file video đầu ra.'
    )
    
    args = parser.parse_args()

    # Ví dụ cách chạy từ dòng lệnh:
    # python merge_clips.py "demo/UEFA_RMA_MCI_10042024/high_quality_95%"
    # python merge_clips.py "demo/WC_ITA_FRA_2006/high_quality_95%" -o "wc_2006_highlights.mp4"

    if not os.path.isdir(args.input_dir):
        print(f"LỖI: Thư mục đầu vào không tồn tại: {args.input_dir}")
    else:
        merge_videos(args.input_dir, args.output) 