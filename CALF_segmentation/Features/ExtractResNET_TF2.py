import argparse
import os
import SoccerNet



import configparser
import math
try:
    from tensorflow.keras.models import Model  # pip install tensorflow (==2.3.0)
    from tensorflow.keras.applications.resnet import preprocess_input
    # from tensorflow.keras.preprocessing.image import img_to_array
    # from tensorflow.keras.preprocessing.image import load_img
    from tensorflow import keras
except:
    print("issue loading TF2")
    pass

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
except:
    print("issue loading PT")
    torch = None

# Import NVDEC frame loader
try:
    from FrameNVDEC import FrameNVDEC, is_nvdec_available, create_frame_loader, get_frames_from_loader
    NVDEC_AVAILABLE = True
except ImportError:
    FrameNVDEC = None
    is_nvdec_available = lambda: False
    create_frame_loader = None
    get_frames_from_loader = None
    NVDEC_AVAILABLE = False
    print("FrameNVDEC not available; NVDEC grabber will be disabled.")

import os
# import argparse
import numpy as np
import cv2  # pip install opencv-python (==3.4.11.41)
import imutils  # pip install imutils
import skvideo.io
from tqdm import tqdm

import json

import random
from SoccerNet.utils import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.DataLoader import Frame, FrameCV


class FeatureExtractor():
    def __init__(self, rootFolder,
                 feature="ResNET",
                 video="LQ",
                 back_end="PT",
                 overwrite=False,
                 transform="crop",
                 tmp_HQ_videos=None,
                 grabber="opencv",
                 FPS=2.0,
                 split="all"):
        self.rootFolder = rootFolder
        self.feature = feature
        self.video = video
        self.back_end = back_end
        self.verbose = True
        self.transform = transform
        self.overwrite = overwrite
        self.grabber = grabber
        self.FPS = FPS
        self.split = split

        self.tmp_HQ_videos = tmp_HQ_videos
        if self.tmp_HQ_videos:
            self.mySoccerNetDownloader = SoccerNetDownloader(self.rootFolder)
            self.mySoccerNetDownloader.password = self.tmp_HQ_videos

        if "TF2" in self.back_end:

            # create pretrained encoder (here ResNet152, pre-trained on ImageNet)
            base_model = keras.applications.resnet.ResNet152(include_top=True,
                                                             weights='imagenet',
                                                             input_tensor=None,
                                                             input_shape=None,
                                                             pooling=None,
                                                             classes=1000)

            # define model with output after polling layer (dim=2048)
            self.model = Model(base_model.input,
                               outputs=[base_model.get_layer("avg_pool").output])
            self.model.trainable = False

        elif "PT" in self.back_end:
            self.device = torch.device("cuda")
            self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
            self.extraction_layer = self.model.avgpool
            self.model.eval()
            self.model = self.model.to(self.device)

            # Standard PyTorch normalization for ImageNet
            self.transform_PT = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def extractAllGames(self):
        list_game = getListGames(self.split)
        for i_game, game in enumerate(tqdm(list_game)):
            try:
                self.extractGameIndex(i_game)
            except:
                print(f"issue with game {i_game}, {game}")

    def extractGameIndex(self, index):
        print(getListGames(self.split)[index])
        if self.video =="LQ":
            for vid in ["1.mkv","2.mkv"]:
                self.extract(video_path=os.path.join(self.rootFolder, getListGames(self.split)[index], vid))

        elif self.video == "HQ":
            
            # read config for raw HD video
            config = configparser.ConfigParser()
            if not os.path.exists(os.path.join(self.rootFolder, getListGames(self.split)[index], "video.ini")) and self.tmp_HQ_videos is not None:
                self.mySoccerNetDownloader.downloadVideoHD(
                    game=getListGames(self.split)[index], file="video.ini")
            config.read(os.path.join(self.rootFolder, getListGames(self.split)[index], "video.ini"))

            # lopp over videos
            for vid in config.sections():
                video_path = os.path.join(self.rootFolder, getListGames(self.split)[index], vid)

                # cehck if already exists, then skip
                feature_path = video_path[:-4] + f"_{self.feature}_{self.back_end}.npy"
                if os.path.exists(feature_path) and not self.overwrite:
                    print("already exists, early skip")
                    continue

                #Download video if does not exist, but remove it afterwards
                remove_afterwards = False
                if not os.path.exists(video_path) and self.tmp_HQ_videos is not None:
                    remove_afterwards = True
                    self.mySoccerNetDownloader.downloadVideoHD(game=getListGames(self.split)[index], file=vid)

                # extract feature for video
                self.extract(video_path=video_path,
                            start=float(config[vid]["start_time_second"]), 
                            duration=float(config[vid]["duration_second"]))
                
                # remove video if not present before
                if remove_afterwards:
                    os.remove(video_path)

    def extract(self, video_path, start=None, duration=None):
        print("extract video", video_path, "from", start, duration)
        # feature_path = video_path.replace(
        #     ".mkv", f"_{self.feature}_{self.back_end}.npy")
        feature_path = video_path[:-4] + f"_{self.feature}_{self.back_end}.npy"

        if os.path.exists(feature_path) and not self.overwrite:
            return
        
        features = None
        if "TF2" in self.back_end:

            # Create video loader using factory function
            if NVDEC_AVAILABLE and create_frame_loader:
                videoLoader = create_frame_loader(video_path, grabber=self.grabber,
                                                FPS=self.FPS, transform=self.transform,
                                                start=start, duration=duration)
            else:
                # Fallback to traditional loaders
                if self.grabber=="skvideo":
                    videoLoader = Frame(video_path, FPS=self.FPS, transform=self.transform, start=start, duration=duration)
                elif self.grabber=="opencv":
                    videoLoader = FrameCV(video_path, FPS=self.FPS, transform=self.transform, start=start, duration=duration)
                else:
                    print(f"Grabber {self.grabber} not supported without NVDEC; falling back to OpenCV.")
                    videoLoader = FrameCV(video_path, FPS=self.FPS, transform=self.transform, start=start, duration=duration)

            # Get frames using unified interface
            try:
                if NVDEC_AVAILABLE and get_frames_from_loader:
                    raw_frames = get_frames_from_loader(videoLoader)
                else:
                    # Traditional interface
                    raw_frames = videoLoader.frames if hasattr(videoLoader, 'frames') else []

                # Check if frames are available
                if len(raw_frames) == 0:
                    print(f"Warning: No frames found in video {video_path}")
                    return

                # create numpy array (nb_frames x 224 x 224 x 3)
                frames = preprocess_input(raw_frames)

                if duration is None:
                    duration = videoLoader.time_second
                    # time_second = duration
                if self.verbose:
                    print("frames", frames.shape, "fps=", frames.shape[0]/duration)

                # predict the features from the frames (adjust batch size for smaller GPU)
                features = self.model.predict(frames, batch_size=64, verbose=1)
                if self.verbose:
                    print("features", features.shape, "fps=", features.shape[0]/duration)

            except Exception as e:
                print(f"Error processing frames with TF2: {e}")
                return



        elif "PT" in self.back_end:
            # Create video loader using factory function
            if NVDEC_AVAILABLE and create_frame_loader:
                videoLoader = create_frame_loader(video_path, grabber=self.grabber,
                                                FPS=self.FPS, transform=self.transform,
                                                start=start, duration=duration)
            else:
                # Fallback to traditional loaders
                if self.grabber=="skvideo":
                    videoLoader = Frame(video_path, FPS=self.FPS, transform=self.transform, start=start, duration=duration)
                elif self.grabber=="opencv":
                    videoLoader = FrameCV(video_path, FPS=self.FPS, transform=self.transform, start=start, duration=duration)
                else:
                    print(f"Grabber {self.grabber} not supported without NVDEC; falling back to OpenCV.")
                    videoLoader = FrameCV(video_path, FPS=self.FPS, transform=self.transform, start=start, duration=duration)

            if duration is None:
                duration = videoLoader.time_second

            # Get frames using unified interface
            try:
                if NVDEC_AVAILABLE and get_frames_from_loader:
                    raw_frames = get_frames_from_loader(videoLoader)
                else:
                    # Traditional interface
                    raw_frames = videoLoader.frames if hasattr(videoLoader, 'frames') else []

                # Check if frames are available
                if len(raw_frames) == 0:
                    print(f"Warning: No frames found in video {video_path}")
                    return

                all_features = []

                # Hook function to get the output from the avgpool layer
                def hook(module, input, output):
                    all_features.append(output.cpu().numpy().squeeze())

                handle = self.extraction_layer.register_forward_hook(hook)

                # Process frames in batches to avoid OOM
                batch_size = 64
                for i in tqdm(range(0, len(raw_frames), batch_size)):
                    frame_batch = raw_frames[i:i+batch_size]

                    # Convert BGR (from cv2) to RGB and apply transform
                    # For NVDEC frames are already RGB, for OpenCV they are BGR
                    if NVDEC_AVAILABLE and isinstance(videoLoader, FrameNVDEC):
                        # NVDEC frames are already RGB numpy arrays
                        frame_batch_transformed = [self.transform_PT(frame) for frame in frame_batch]
                    else:
                        # OpenCV frames are BGR, convert to RGB
                        frame_batch_transformed = [self.transform_PT(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frame_batch]

                    batch = torch.stack(frame_batch_transformed, dim=0).to(self.device)

                    with torch.no_grad():
                        self.model(batch)

                handle.remove()
                # Concatenate batch outputs to shape (num_frames, 2048)
                if len(all_features) > 0:
                    features = np.concatenate(all_features, axis=0)
                else:
                    features = np.empty((0, 2048), dtype=np.float32)

            except Exception as e:
                print(f"Error processing frames: {e}")
                return
            
            if self.verbose:
                print("features", features.shape, "fps=", features.shape[0]/duration)

        # save the featrue in .npy format
        if features is not None:
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
            np.save(feature_path, features)
        else:
            raise ValueError(f"Unsupported backend: {self.back_end}. Features could not be extracted.")



if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Extract ResNet feature out of SoccerNet Videos.')

    parser.add_argument('--soccernet_dirpath', type=str, default="/media/giancos/Football/SoccerNet/",
                        help="Path for SoccerNet directory [default:/media/giancos/Football/SoccerNet/]")
                        
    parser.add_argument('--overwrite', action="store_true",
                        help="Overwrite the features? [default:False]")
    parser.add_argument('--GPU', type=int, default=0,
                        help="ID of the GPU to use [default:0]")
    parser.add_argument('--verbose', action="store_true",
                        help="Print verbose? [default:False]")
    parser.add_argument('--game_ID', type=int, default=None,
                        help="ID of the game from which to extract features. If set to None, then loop over all games. [default:None]")

    # feature setup
    parser.add_argument('--back_end', type=str, default="PT",
                        help="Backend TF2 or PT [default:PT]")
    parser.add_argument('--features', type=str, default="ResNET",
                        help="ResNET or R25D [default:ResNET]")
    parser.add_argument('--transform', type=str, default="crop",
                        help="crop or resize? [default:crop]")
    parser.add_argument('--video', type=str, default="LQ",
                        help="LQ or HQ? [default:LQ]")
    parser.add_argument('--grabber', type=str, default="opencv",
                        help="skvideo, opencv, or nvdec? [default:opencv]")
    parser.add_argument('--tmp_HQ_videos', type=str, default=None,
                        help="enter pawssword to download and store temporally the videos [default:None]")
    parser.add_argument('--FPS', type=float, default=2.0,
                        help="FPS for the features [default:2.0]")
    parser.add_argument('--split', type=str, default="all",
                        help="split of videos from soccernet [default:all]")

    args = parser.parse_args()
    print(args)

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    myFeatureExtractor = FeatureExtractor(
        args.soccernet_dirpath, 
        feature=args.features,
        video=args.video,
        back_end=args.back_end, 
        # array=args.array,
        transform=args.transform,
        # preprocess=True,
        tmp_HQ_videos=args.tmp_HQ_videos,
        grabber=args.grabber,
        FPS=args.FPS,
        split=args.split)
    myFeatureExtractor.overwrite= args.overwrite

    # def extractGameIndex(self, index):
    if args.game_ID is None:
        myFeatureExtractor.extractAllGames()
    else:
        myFeatureExtractor.extractGameIndex(args.game_ID)
