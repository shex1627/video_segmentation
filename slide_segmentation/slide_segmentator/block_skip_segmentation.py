import logging
import sys
import os

import cv2

from slide_segmentation.ocr.config import new_slide_clf
from slide_segmentation.util import *
from slide_segmentation.ocr.ocr_process import *
from slide_segmentation.ocr.blockSkipSegmentator import BlockSkipSegmentator


formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s')
logging.getLogger("shapely.geos").setLevel(logging.INFO)
logger = logging.getLogger() #'block_skip_segmentation_main'
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(filename='DEBUG_block_skip_segmentation_main.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

info_file_handler = logging.FileHandler(filename='INFO_block_skip_segmentation_main.log', mode='w')
info_file_handler.setLevel(logging.INFO)
info_file_handler.setFormatter(formatter)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(info_file_handler)
logger.addHandler(stdout_handler)


def main():
    # create img temp dir if not exist
    logger.debug("this msg only exists in debug mode")
    img_tmp_dir = args.img_sample_dir 
    video_file_path = args.input_file 
    video_file_name = video_file_path.split("\\")[-1].split(".")[0]
    ocr_output_dir = os.path.join(img_tmp_dir, "video_ocr_results")
    print(video_file_name)
    mkdir_if_not_exist(img_tmp_dir)
    output_data_dir = os.path.join(img_tmp_dir, video_file_name)
    mkdir_if_not_exist(output_data_dir)
    mkdir_if_not_exist(ocr_output_dir)
    # sample img, estimate fps
    logger.info("sampling images")
    video_fps = get_fps(video_file_path) # doubts if this is right
    logger.info(f"video fps {video_fps}")
    sample_video_img(video_file_path, output_data_dir, seconds_per_frame=1, fps=video_fps, max_duration=args.max_duration)

    img_files = list(Path(output_data_dir).glob("*.jpg"))
    img_files = sorted(img_files, key=lambda file: int(file.name.split(".")[0]))
    img_file_dict = {int(file.name.split(".")[0]): cv2.imread(str(file), cv2.IMREAD_GRAYSCALE) 
                 for file in img_files}
    
    ocr = init_ocr()
    segmentator = BlockSkipSegmentator(chunk_size=10, img_file_dict=img_file_dict, ocr_model=ocr, clf_model=new_slide_clf)
    segmentation_result = segmentator.get_slide_breakpoints(max_block_count=10)


    ocf_features_dir = os.path.join(img_tmp_dir, "video_ocf_features")
    mkdir_if_not_exist(ocf_features_dir)
    segmentation_result.to_csv(os.path.join(ocf_features_dir, video_file_name+"_new_slide.csv"), index=False)
    # # run ocr
    # ocr_results = run_ocr(output_data_dir, ocr_output_dir)
    # # process ocr and get diff df
    # diff_df = get_diff_df(ocr_results)
    # ocf_features_dif = os.path.join(img_tmp_dir, "video_ocf_features")
    # mkdir_if_not_exist(ocf_features_dif)
    # diff_df.to_csv(os.path.join( ocf_features_dif, video_file_name+".csv"), index=False)

    # # load model and classify
    # prediction_df = diff_df.copy()
    # prediction_df = prediction_df.rename(columns={'letter_dis':'letter_dissim'})
    # prediction_df['dissimilarity'] = 1
    # prediction_df['new_slide_prediction'] = new_slide_clf.predict(prediction_df[feature_names])
    # prediction_df.query("new_slide_prediction").to_csv(os.path.join(ocf_features_dif, video_file_name+"_new_slide.csv"), index=False)

    return segmentation_result


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="input mp4 file path")
    parser.add_argument("--img_sample_dir", help="directory to store the images and outputs", default=".")
    parser.add_argument("--max_duration", help="max number of seconds of the video to process", default=float("inf"), type=int)
    parser.add_argument("--max_block_count", help="max number of blocks to process", default=float("inf"), type=int)
    args = parser.parse_args()
    main()