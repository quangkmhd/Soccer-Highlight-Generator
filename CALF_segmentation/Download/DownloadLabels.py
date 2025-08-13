#!/usr/bin/env python3
"""
Script để tải labels cho camera segmentation task từ SoccerNet
Sử dụng khi đã có video nhưng thiếu labels
"""

import os
import sys
import json
from pathlib import Path
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

def check_existing_structure(soccernet_path):
    """Kiểm tra cấu trúc thư mục hiện tại"""
    print("=== CHECKING EXISTING STRUCTURE ===")
    
    if not os.path.exists(soccernet_path):
        print(f"❌ SoccerNet directory not found: {soccernet_path}")
        return False
    
    splits = ["train", "valid", "test", "challenge"]
    total_games = 0
    games_with_videos = 0
    games_with_labels = 0
    
    for split in splits:
        split_path = os.path.join(soccernet_path, split)
        if os.path.exists(split_path):
            games = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            print(f"\n📁 {split}: {len(games)} games")
            
            for game in games[:3]:  # Check first 3 games as sample
                game_path = os.path.join(split_path, game)
                total_games += 1
                
                # Check videos
                video1 = os.path.join(game_path, "1_224p.mkv")
                video2 = os.path.join(game_path, "2_224p.mkv")
                has_videos = os.path.exists(video1) and os.path.exists(video2)
                
                # Check labels
                labels = os.path.join(game_path, "Labels-cameras.json")
                has_labels = os.path.exists(labels)
                
                if has_videos:
                    games_with_videos += 1
                if has_labels:
                    games_with_labels += 1
                
                status = "✅" if has_videos and has_labels else "⚠️" if has_videos else "❌"
                print(f"  {status} {game}: Videos={has_videos}, Labels={has_labels}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total games checked: {total_games}")
    print(f"   Games with videos: {games_with_videos}")
    print(f"   Games with labels: {games_with_labels}")
    print(f"   Missing labels: {games_with_videos - games_with_labels}")
    
    return games_with_videos > 0

def download_labels_only(soccernet_path, password=None, splits=None):
    """Tải chỉ labels cho camera segmentation task"""
    print("\n=== DOWNLOADING LABELS ===")
    
    if splits is None:
        splits = ["train", "valid", "test", "challenge"]
    
    # Khởi tạo downloader
    downloader = SoccerNetDownloader(LocalDirectory=soccernet_path)
    
    if password:
        downloader.password = password
    else:
        downloader.password = input("Enter SoccerNet password: ")
    
    # Chỉ tải labels
    files_to_download = ["Labels-cameras.json"]
    
    print(f"📥 Downloading files: {files_to_download}")
    print(f"📁 For splits: {splits}")
    
    try:
        downloader.downloadGames(
            files=files_to_download,
            split=splits,
            task="camera-changes",  # Quan trọng: chỉ định task
            verbose=True
        )
        print("✅ Labels download completed!")
        return True
    except Exception as e:
        print(f"❌ Error downloading labels: {e}")
        return False

def verify_labels_structure(soccernet_path):
    """Kiểm tra cấu trúc labels sau khi tải"""
    print("\n=== VERIFYING LABELS STRUCTURE ===")
    
    splits = ["train", "valid", "test", "challenge"]
    total_checked = 0
    valid_labels = 0
    
    for split in splits:
        split_path = os.path.join(soccernet_path, split)
        if not os.path.exists(split_path):
            continue
            
        games = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        
        for game in games[:2]:  # Check first 2 games per split
            labels_path = os.path.join(split_path, game, "Labels-cameras.json")
            total_checked += 1
            
            if os.path.exists(labels_path):
                try:
                    with open(labels_path, 'r') as f:
                        labels_data = json.load(f)
                    
                    # Kiểm tra cấu trúc cơ bản
                    if "annotations" in labels_data:
                        annotations = labels_data["annotations"]
                        print(f"✅ {split}/{game}: {len(annotations)} annotations")
                        
                        # Hiển thị sample annotation
                        if annotations:
                            sample = annotations[0]
                            print(f"   Sample: {sample.get('gameTime', 'N/A')} - {sample.get('label', 'N/A')} - {sample.get('change_type', 'N/A')}")
                        
                        valid_labels += 1
                    else:
                        print(f"❌ {split}/{game}: Invalid structure")
                        
                except Exception as e:
                    print(f"❌ {split}/{game}: Error reading - {e}")
            else:
                print(f"❌ {split}/{game}: Labels not found")
    
    print(f"\n📊 VERIFICATION SUMMARY:")
    print(f"   Total checked: {total_checked}")
    print(f"   Valid labels: {valid_labels}")
    print(f"   Success rate: {valid_labels/total_checked*100:.1f}%" if total_checked > 0 else "   No games to check")

def main():
    """Main function"""
    print("🏈 SoccerNet Labels Downloader for Camera Segmentation")
    print("=" * 60)
    
    # Đường dẫn SoccerNet
    soccernet_path = "CALF_segmentation/SoccerNet"
    
    # Kiểm tra cấu trúc hiện tại
    if not check_existing_structure(soccernet_path):
        print("❌ No existing structure found. Please run DownloadSoccerNet.py first.")
        return
    
    # Hỏi user có muốn tải labels không
    response = input("\n🤔 Do you want to download labels? (y/n): ").lower().strip()
    if response != 'y':
        print("👋 Exiting...")
        return
    
    # Chọn splits để tải
    print("\n📁 Available splits: train, valid, test, challenge")
    splits_input = input("Enter splits to download (comma-separated, or 'all'): ").strip()
    
    if splits_input.lower() == 'all':
        splits = ["train", "valid", "test", "challenge"]
    else:
        splits = [s.strip() for s in splits_input.split(',') if s.strip()]
    
    if not splits:
        print("❌ No valid splits specified")
        return
    
    # Tải labels
    success = download_labels_only(soccernet_path, splits=splits)
    
    if success:
        # Kiểm tra kết quả
        verify_labels_structure(soccernet_path)
        
        print("\n🎉 DOWNLOAD COMPLETED!")
        print("📁 Your structure should now be:")
        print("CALF_segmentation/SoccerNet/")
        print("├── train/")
        print("├── valid/")
        print("├── test/")
        print("└── challenge/")
        print("    └── [game_folders]/")
        print("        ├── 1_224p.mkv")
        print("        ├── 2_224p.mkv")
        print("        └── Labels-cameras.json  ← NEW!")
        
        print("\n🚀 Next steps:")
        print("1. Extract features: python CALF_segmentation/Features/ExtractResNET_TF2.py")
        print("2. Train model: python CALF_segmentation/src/main.py")
    else:
        print("❌ Download failed. Please check your credentials and try again.")

if __name__ == "__main__":
    main()
