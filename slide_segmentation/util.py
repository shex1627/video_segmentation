from cmath import inf
import cv2
import logging
import os
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger("util")

def get_fps(video_path: str):
    """find the fps of a video using opencv get cap prop fps"""
    video = cv2.VideoCapture(video_path)

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        logger.info("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        logger.info("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    video.release()

    return int(fps)


def mkdir_if_not_exist(path: str):
    isdir = os.path.isdir(path) 
    if not isdir:
        os.mkdir(path)
    return 


def sample_video_img(video_file: str, output_data_dir: str, seconds_per_frame: int=1, fps: int=24,
                max_duration=float('inf')):
    """sample videos into frames."""
    # check if exist 
    max_i = max_duration * fps
    logger.info(f"max_i_counter: {max_i}")
    cap= cv2.VideoCapture(str(video_file))
    i=0
    while(cap.isOpened() and i <= max_i):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % (fps*seconds_per_frame) == 0: # this is the line I added to make it only save one frame every 20
            time = int(i/fps)
            cv2.imwrite(os.path.join(output_data_dir, str(time)+'.jpg'),frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()
    return 0


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def img_resize(img, scale: int=60):
        scale_percent = 60 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized

def init_ocr():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    return ocr
    

def run_ocr(img_file_dir: str, output_file_dir: str="../data/video_ocr_results"):
    ocr = init_ocr()

    img_files = list(Path(img_file_dir).glob("*.jpg"))
    video_name = img_file_dir.split("\\")[-1]
    img_file_dict = {int(file.name.split(".")[0]): cv2.imread(str(file), cv2.IMREAD_GRAYSCALE) 
                 for file in img_files}

    img_ocr_dict = {key: ocr.ocr(frame_img) for key, frame_img in img_file_dict.items()}
    if output_file_dir:
        with open(os.path.join(output_file_dir, f"{video_name}.json"), 'w') as file:
            json.dump(img_ocr_dict, file, cls=NpEncoder, indent=4)
    return img_ocr_dict

