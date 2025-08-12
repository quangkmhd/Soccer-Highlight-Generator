import argparse
import os
import logging

import numpy as np
import cv2
import pickle as pkl
from SoccerNet.DataLoader import Frame, FrameCV

try:
    from tensorflow.keras.models import Model as TFModel
    from tensorflow.keras.applications.resnet import preprocess_input as tf_preprocess
    from tensorflow import keras
    import tensorflow as tf
except Exception:
    logging.warning("TensorFlow not available; TF2 backend will fail.")


class VideoFeatureExtractor:
    def __init__(self,
                 feature: str = "ResNET",
                 back_end: str = "TF2",
                 overwrite: bool = False,
                 transform: str = "crop",
                 grabber: str = "opencv",
                 FPS: float = 2.0,
                 split: str = "all"):

        self.feature = feature
        self.back_end = "TF2"
        self.verbose = True
        self.transform = transform
        self.overwrite = overwrite
        self.grabber = grabber
        self.FPS = FPS
        self.split = split

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logging.warning("TF GPU config error: %s", e)

        base_model = keras.applications.resnet.ResNet152(
            include_top=True,
            weights='/home/qmask_quangnh58/soccer_highlight/CALF_segmentation/Features/resnet152_weights_tf_dim_ordering_tf_kernels.h5',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000)
        self.model = TFModel(base_model.input,
                             outputs=[base_model.get_layer("avg_pool").output])
        self.model.trainable = False

    def resize_with_padding(self, frame, target_size=(224, 224)):
        h, w = frame.shape[:2]
        target_h, target_w = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        return canvas

    def process_frames(self, frames, target_size=(224, 224)):
        processed = []
        for f in frames:
            processed.append(self.resize_with_padding(f, target_size))
        return np.array(processed)

    def extractFeatures(self, path_video_input, path_features_output,
                        start=None, duration=None, overwrite=False):
        logging.info(f"Extracting features for video {path_video_input}")
        if os.path.exists(path_features_output) and not overwrite:
            logging.info("Features already exist, use overwrite=True to overwrite.")
            return

        if self.grabber == "skvideo":
            videoLoader = Frame(path_video_input, FPS=self.FPS,
                                transform=self.transform, start=start, duration=duration)
        elif self.grabber == "opencv":
            videoLoader = FrameCV(path_video_input, FPS=self.FPS,
                                  transform=self.transform, start=start, duration=duration)
        else:
            raise ValueError(f"Unsupported grabber: {self.grabber}")

        raw_frames = videoLoader.frames
        processed = self.process_frames(raw_frames, target_size=(224, 224))
        frames_tf = tf_preprocess(processed)

        features = self.model.predict(frames_tf, batch_size=64, verbose=1)
        np.save(path_features_output, features)

        logging.info(f"Saved features to {path_features_output} with shape {features.shape}")


class PCAReducer:
    def __init__(self, pca_file=None, scaler_file=None):
        self.pca_file = pca_file
        self.scaler_file = scaler_file
        self.loadPCA()

    def loadPCA(self):
        self.pca = None
        if self.pca_file is not None:
            with open(self.pca_file, "rb") as fobj:
                self.pca = pkl.load(fobj)
        self.average = None
        if self.scaler_file is not None:
            with open(self.scaler_file, "rb") as fobj:
                self.average = pkl.load(fobj)

    def reduceFeatures(self, input_features, output_features, overwrite=False):
        logging.info(f"Reducing features {input_features}")
        if os.path.exists(output_features) and not overwrite:
            logging.info("Features already exist, use overwrite=True to overwrite.")
            return
        feat = np.load(input_features)
        if self.average is not None:
            feat = feat - self.average
        if self.pca is not None:
            feat = self.pca.transform(feat)
        np.save(output_features, feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract ResNet feature from a video (TF2 backend).')

    parser.add_argument('--path_video', type=str, required=True,
                        help="Path of the Input Video")
    parser.add_argument('--path_features', type=str, required=True,
                        help="Path of the Output Features")
    parser.add_argument('--start', type=float, default=None,
                        help="time of the video to start extracting features [default:None]")
    parser.add_argument('--duration', type=float, default=None,
                        help="duration of the video before finishing extracting features [default:None]")
    parser.add_argument('--overwrite', action="store_true",
                        help="Overwrite the features.")

    parser.add_argument('--GPU', type=int, default=0,
                        help="ID of the GPU to use [default:0]")
    parser.add_argument('--loglevel', type=str, default="INFO",
                        help="loglevel for logging [default:INFO]")

    parser.add_argument('--back_end', type=str, default="TF2",
                        help="Only TF2 is supported after refactor [default:TF2]")
    parser.add_argument('--features', type=str, default="ResNET",
                        help="ResNET or R25D [default:ResNET]")
    parser.add_argument('--transform', type=str, default="crop",
                        help="crop or resize? [default:crop]")
    parser.add_argument('--video', type=str, default="LQ",
                        help="LQ or HQ? [default:LQ]")
    parser.add_argument('--grabber', type=str, default="opencv",
                        help="skvideo or opencv? [default:opencv]")
    parser.add_argument('--FPS', type=float, default=2.0,
                        help="FPS for the features [default:2.0]")

    parser.add_argument('--PCA', type=str, default="pca_512_TF2.pkl",
                        help="Pickle with pre-computed PCA")
    parser.add_argument('--PCA_scaler', type=str, default="average_512_TF2.pkl",
                        help="Pickle with pre-computed PCA scaler")

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), None),
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[logging.StreamHandler()])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    myFeatureExtractor = VideoFeatureExtractor(
        feature=args.features,
        back_end=args.back_end,
        transform=args.transform,
        grabber=args.grabber,
        FPS=args.FPS)

    myFeatureExtractor.extractFeatures(path_video_input=args.path_video,
                                       path_features_output=args.path_features,
                                       start=args.start,
                                       duration=args.duration,
                                       overwrite=args.overwrite)

    if args.PCA is not None or args.PCA_scaler is not None:
        myPCAReducer = PCAReducer(pca_file=args.PCA,
                                  scaler_file=args.PCA_scaler)

        myPCAReducer.reduceFeatures(input_features=args.path_features,
                                    output_features=args.path_features,
                                    overwrite=args.overwrite)
