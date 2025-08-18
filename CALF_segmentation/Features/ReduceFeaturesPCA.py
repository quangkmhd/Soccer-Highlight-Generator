
import argparse

import numpy as np
import os 
from sklearn.decomposition import PCA , IncrementalPCA  # pip install scikit-learn
from sklearn.preprocessing import StandardScaler
from SoccerNet.utils import getListGames
from datetime import datetime
import pickle as pkl

from tqdm import tqdm



def main(args):

    if not os.path.exists(args.pca_file) or not os.path.exists(args.scaler_file):
        
        game_files = []
        # Collect all feature files first
        print("Collecting feature files...")
        for game in tqdm(getListGames(split="v1",task="camera-changes")):
            for half in [1, 2]:
                feature_file = os.path.join(args.soccernet_dirpath, game, f"{half}_224p_{args.features}")
                if os.path.exists(feature_file):
                    game_files.append(feature_file)

        # --- Pass 1: Calculate the mean incrementally to avoid memory issues ---
        print("Pass 1: Calculating mean of features...")
        total_sum = None
        total_count = 0
        for f in tqdm(game_files):
            feat = np.load(f)
            if total_sum is None:
                # Use float64 for sum to maintain precision with large numbers
                total_sum = np.sum(feat, axis=0, dtype=np.float64)
            else:
                total_sum += np.sum(feat, axis=0, dtype=np.float64)
            total_count += feat.shape[0]
        
        average = (total_sum / total_count).astype(np.float32)
        
        # Store average for later
        with open(args.scaler_file, "wb") as fobj:
            pkl.dump(average, fobj)
        print("Mean calculated and saved.")

        # --- Pass 2: Fit IncrementalPCA, which is suitable for large datasets ---
        print("Pass 2: Fitting IncrementalPCA...")
        # Batch size can be tuned based on available RAM, but this is a safe default
        pca = IncrementalPCA(n_components=args.dim_reduction, batch_size=max(2048, args.dim_reduction))
        print(datetime.now(), "IncrementalPCA start")
        for f in tqdm(game_files):
            feat = np.load(f)
            # Center the data with the pre-calculated mean before fitting
            pca.partial_fit(feat - average)
        print(datetime.now(), "IncrementalPCA fitted")

        # Store PCA for later
        with open(args.pca_file, "wb") as fobj:
            pkl.dump(pca, fobj)
        print("PCA model saved.")




    # Read pre-computed PCA
    with open(args.pca_file, "rb") as fobj:
        pca = pkl.load(fobj)

    # Read pre-computed average
    with open(args.scaler_file, "rb") as fobj:
        average = pkl.load(fobj)


    # loop over games in v1
    for game in tqdm(getListGames(split="v1",task="camera-changes")):
        for half in [1,2]:
            game_feat = os.path.join(args.soccernet_dirpath, game, f"{half}_224p_{args.features}")
            game_feat_pca = os.path.join(args.soccernet_dirpath, game, f"{half}_224p_{args.features_PCA}")

            if not os.path.exists(game_feat_pca) or args.overwrite:
                feat = np.load(game_feat)
                feat = feat - average
                feat_reduced = pca.transform(feat)
                np.save(game_feat_pca, feat_reduced)
            else:
                print(f"{game_feat_pca} already exists")


    # for game in tqdm(getListGames(["challenge"])):
    #     for half in [1,2]:
    #         game_feat = os.path.join(args.soccernet_dirpath, game, f"{half}_{args.features}")
    #         game_feat_pca = os.path.join(args.soccernet_dirpath, game, f"{half}_{args.features_PCA}")
    #         if not os.path.exists(game_feat_pca) or args.overwrite:
    #             feat = np.load(game_feat)
                
    #             feat = feat - average
                
    #             feat_reduced = pca.transform(feat)

    #             np.save(game_feat_pca, feat_reduced)

    #         else:
    #             print(f"{game_feat_pca} already exists")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Extract ResNet feature out of SoccerNet Videos.')

    parser.add_argument('--soccernet_dirpath', type=str, default="/media/giancos/Football/SoccerNet/",
                        help="Path for SoccerNet directory [default:/media/giancos/Football/SoccerNet/]")
    parser.add_argument('--features', type=str, default="ResNET_PT.npy",
                        help="features to perform PCA on [default:ResNET_PT.npy]")    
    parser.add_argument('--features_PCA', type=str, default="ResNET_PT_PCA512.npy",
                        help="name of reduced features [default:ResNET_PT_PCA512.npy]")
    parser.add_argument('--pca_file', type=str, default="pca_512_PT.pkl",
                        help="pickle for PCA [default:pca_512_PT.pkl]")
    parser.add_argument('--scaler_file', type=str, default="average_512_PT.pkl",
                        help="pickle for average [default:average_512_PT.pkl]")
    parser.add_argument('--dim_reduction', type=int, default=512,
                        help="dimension reduction [default:512]")

    parser.add_argument('--overwrite', action="store_true",
                        help="Overwrite the features? [default:False]")

    args = parser.parse_args()
    print(args)

    main(args)