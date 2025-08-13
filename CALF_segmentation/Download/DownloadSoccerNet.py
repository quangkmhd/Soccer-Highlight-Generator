import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

# Khởi tạo SoccerNet Downloader
mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="CALF_segmentation/SoccerNet")

# Nhập password (comment dòng này nếu đã có password)
mySoccerNetDownloader.password = input("Password for videos?:\n")

# Hoặc set password trực tiếp (uncomment và thay thế password thật)
# mySoccerNetDownloader.password = "your_actual_password_here"

print("Downloading videos and labels for camera segmentation task...")

# Tải video và labels cho camera segmentation task
# Labels-cameras.json chứa annotations cho camera types và camera changes
files_to_download = [
    "1_224p.mkv",           # Video hiệp 1 (224p resolution)
    "2_224p.mkv",           # Video hiệp 2 (224p resolution)
    "Labels-cameras.json"   # Labels cho camera segmentation task
]

# Tải cho tất cả splits
splits_to_download = ["train", "valid", "test", "challenge"]

print(f"Files to download: {files_to_download}")
print(f"Splits to download: {splits_to_download}")

# Download với task="camera-changes" để đảm bảo tải đúng labels
mySoccerNetDownloader.downloadGames(
    files=files_to_download,
    split=splits_to_download,
    task="camera-changes",  # Quan trọng: chỉ định task để tải đúng labels
    verbose=True
)

print("Download completed!")
print("Structure should be:")
print("CALF_segmentation/SoccerNet/")
print("├── train/")
print("├── valid/")
print("├── test/")
print("└── challenge/")
print("    └── [game_folders]/")
print("        ├── 1_224p.mkv")
print("        ├── 2_224p.mkv")
print("        └── Labels-cameras.json")