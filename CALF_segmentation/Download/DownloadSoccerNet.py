import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="CALF_segmentation/SoccerNet")

mySoccerNetDownloader.password = input("Password for videos?:\n")


print("Downloading videos and labels for camera segmentation task...")

files_to_download = [
    "1_224p.mkv",           
    "2_224p.mkv",           
    "Labels-cameras.json"   
]

splits_to_download = ["train", "valid", "test", "challenge"]

print(f"Files to download: {files_to_download}")
print(f"Splits to download: {splits_to_download}")

mySoccerNetDownloader.downloadGames(
    files=files_to_download,
    split=splits_to_download,
    task="camera-changes",  
    verbose=True
)

