#!/usr/bin/env python3
"""
Script để kiểm tra cấu trúc dữ liệu SoccerNet cho camera segmentation task
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def check_data_structure(soccernet_path="CALF_segmentation/SoccerNet"):
    """Kiểm tra cấu trúc dữ liệu chi tiết"""
    print("🔍 CHECKING SOCCERNET DATA STRUCTURE")
    print("=" * 60)
    
    if not os.path.exists(soccernet_path):
        print(f"❌ SoccerNet directory not found: {soccernet_path}")
        return False
    
    splits = ["train", "valid", "test", "challenge"]
    stats = defaultdict(lambda: defaultdict(int))
    
    for split in splits:
        split_path = os.path.join(soccernet_path, split)
        print(f"\n📁 Checking {split.upper()} split...")
        
        if not os.path.exists(split_path):
            print(f"   ❌ Split directory not found: {split_path}")
            continue
        
        games = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        stats[split]['total_games'] = len(games)
        
        print(f"   📊 Found {len(games)} games")
        
        for i, game in enumerate(games):
            game_path = os.path.join(split_path, game)
            
            # Check videos
            video1_path = os.path.join(game_path, "1_224p.mkv")
            video2_path = os.path.join(game_path, "2_224p.mkv")
            has_video1 = os.path.exists(video1_path)
            has_video2 = os.path.exists(video2_path)
            
            # Check labels
            labels_path = os.path.join(game_path, "Labels-cameras.json")
            has_labels = os.path.exists(labels_path)
            
            # Update stats
            if has_video1:
                stats[split]['video1_count'] += 1
            if has_video2:
                stats[split]['video2_count'] += 1
            if has_labels:
                stats[split]['labels_count'] += 1
            if has_video1 and has_video2 and has_labels:
                stats[split]['complete_games'] += 1
            
            # Show detailed info for first few games
            if i < 5:
                status_video1 = "✅" if has_video1 else "❌"
                status_video2 = "✅" if has_video2 else "❌"
                status_labels = "✅" if has_labels else "❌"
                
                print(f"   {game}:")
                print(f"     {status_video1} 1_224p.mkv")
                print(f"     {status_video2} 2_224p.mkv")
                print(f"     {status_labels} Labels-cameras.json")
                
                # Check labels content if exists
                if has_labels:
                    try:
                        with open(labels_path, 'r') as f:
                            labels_data = json.load(f)
                        annotations = labels_data.get('annotations', [])
                        print(f"       📝 {len(annotations)} annotations")
                        
                        # Sample annotation
                        if annotations:
                            sample = annotations[0]
                            print(f"       📄 Sample: {sample.get('gameTime', 'N/A')} - {sample.get('label', 'N/A')}")
                    except Exception as e:
                        print(f"       ❌ Error reading labels: {e}")
        
        if len(games) > 5:
            print(f"   ... and {len(games) - 5} more games")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY STATISTICS")
    print("=" * 60)
    
    total_games = 0
    total_complete = 0
    
    for split in splits:
        if stats[split]['total_games'] > 0:
            total_games += stats[split]['total_games']
            total_complete += stats[split]['complete_games']
            
            print(f"\n{split.upper()}:")
            print(f"  📁 Total games: {stats[split]['total_games']}")
            print(f"  🎬 Video 1: {stats[split]['video1_count']}")
            print(f"  🎬 Video 2: {stats[split]['video2_count']}")
            print(f"  📝 Labels: {stats[split]['labels_count']}")
            print(f"  ✅ Complete: {stats[split]['complete_games']}")
            
            if stats[split]['total_games'] > 0:
                completion_rate = stats[split]['complete_games'] / stats[split]['total_games'] * 100
                print(f"  📈 Completion: {completion_rate:.1f}%")
    
    print(f"\n🌍 OVERALL:")
    print(f"  📁 Total games: {total_games}")
    print(f"  ✅ Complete games: {total_complete}")
    if total_games > 0:
        overall_completion = total_complete / total_games * 100
        print(f"  📈 Overall completion: {overall_completion:.1f}%")
    
    return total_complete > 0

def check_labels_content(soccernet_path="CALF_segmentation/SoccerNet", max_games=3):
    """Kiểm tra nội dung labels chi tiết"""
    print("\n" + "=" * 60)
    print("🔍 CHECKING LABELS CONTENT")
    print("=" * 60)
    
    splits = ["train", "valid", "test"]
    camera_types = defaultdict(int)
    change_types = defaultdict(int)
    
    for split in splits:
        split_path = os.path.join(soccernet_path, split)
        if not os.path.exists(split_path):
            continue
        
        games = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        
        print(f"\n📁 {split.upper()} - Checking {min(max_games, len(games))} games...")
        
        for game in games[:max_games]:
            labels_path = os.path.join(split_path, game, "Labels-cameras.json")
            
            if os.path.exists(labels_path):
                try:
                    with open(labels_path, 'r') as f:
                        labels_data = json.load(f)
                    
                    annotations = labels_data.get('annotations', [])
                    print(f"  📄 {game}: {len(annotations)} annotations")
                    
                    # Analyze annotations
                    for ann in annotations:
                        camera_type = ann.get('label', 'Unknown')
                        change_type = ann.get('change_type', 'Unknown')
                        camera_types[camera_type] += 1
                        change_types[change_type] += 1
                    
                    # Show sample annotations
                    if annotations:
                        print(f"    📝 Sample annotations:")
                        for i, ann in enumerate(annotations[:3]):
                            time = ann.get('gameTime', 'N/A')
                            label = ann.get('label', 'N/A')
                            change = ann.get('change_type', 'N/A')
                            print(f"      {i+1}. {time} - {label} ({change})")
                        
                        if len(annotations) > 3:
                            print(f"      ... and {len(annotations) - 3} more")
                
                except Exception as e:
                    print(f"  ❌ {game}: Error reading labels - {e}")
            else:
                print(f"  ❌ {game}: Labels not found")
    
    # Print label statistics
    if camera_types:
        print(f"\n📊 CAMERA TYPES DISTRIBUTION:")
        for camera_type, count in sorted(camera_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {camera_type}: {count}")
    
    if change_types:
        print(f"\n📊 CHANGE TYPES DISTRIBUTION:")
        for change_type, count in sorted(change_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {change_type}: {count}")

def main():
    """Main function"""
    print("🏈 SoccerNet Data Structure Checker")
    print("For Camera Segmentation Task")
    
    soccernet_path = "CALF_segmentation/SoccerNet"
    
    # Check basic structure
    has_data = check_data_structure(soccernet_path)
    
    if has_data:
        # Check labels content
        check_labels_content(soccernet_path)
        
        print("\n" + "=" * 60)
        print("🎯 RECOMMENDATIONS")
        print("=" * 60)
        
        print("✅ If you have complete data:")
        print("   1. Extract features: python CALF_segmentation/Features/ExtractResNET_TF2.py")
        print("   2. Train model: python CALF_segmentation/src/main.py")
        
        print("\n⚠️ If you're missing data:")
        print("   1. Missing videos: python CALF_segmentation/Download/DownloadSoccerNet.py")
        print("   2. Missing labels: python CALF_segmentation/Download/DownloadLabels.py")
        
    else:
        print("\n❌ No data found. Please download data first:")
        print("   python CALF_segmentation/Download/DownloadSoccerNet.py")

if __name__ == "__main__":
    main()
