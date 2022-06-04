import logging
import sys
import os
from slide_segmentation.util import *
from slide_segmentation.ocr.ocr_process import *
from joblib import dump, load
MODEL_PATH = 'C:\\Users\\alistar\\Desktop\\ds\\video_segmentation\\models\\ocr_tree.joblib'
feature_names = ['dissimilarity', 'jaccard', 'jaccard_letter', 'frame_token_ct',
       'word_dis', 'letter_dissim']
new_slide_clf = load(MODEL_PATH)


file_handler = logging.FileHandler(filename='tmp.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    handlers=handlers
)
logger = logging.getLogger('video_segmentation_main')


def main():
    # create img temp dir if not exist
    img_tmp_dir = args.img_sample_dir #"C:\\Users\\alistar\\Desktop\\ds\\video_segmentation\\data\\tmp" #"./tmp"
    video_file_path = args.input_file #"C:\\Users\\alistar\\Desktop\\ds\\video_segmentation\\data\\test_videos\\test_video_20.mp4"
    video_file_name = video_file_path.split("\\")[-1].split(".")[0]
    ocr_output_dir = "./tmp/video_ocr_results"
    print(video_file_name)
    mkdir_if_not_exist(img_tmp_dir)
    output_data_dir = os.path.join(img_tmp_dir, video_file_name)
    mkdir_if_not_exist(output_data_dir)
    mkdir_if_not_exist(ocr_output_dir)
    # sample img, estimate fps
    logger.info("sampling images")
    video_fps = get_fps(video_file_path)
    logger.info(f"video fps {video_fps}")
    sample_video_img(video_file_path, output_data_dir, seconds_per_frame=1, fps=video_fps, max_duration=args.max_duration)

    # run ocr
    #ocr_results = run_ocr(output_data_dir, ocr_output_dir)
    # process ocr and get diff df
    #diff_df = get_diff_df(ocr_results)
    #ocf_features_dif = os.path.join(img_tmp_dir, "video_ocf_features")
    #mkdir_if_not_exist(ocf_features_dif)
    #diff_df.to_csv(os.path.join( ocf_features_dif, video_file_name+".csv"), index=False)

    # load model and classify
    #prediction_df = diff_df.copy()
    #prediction_df = prediction_df.rename(columns={'letter_dis':'letter_dissim'})
    #prediction_df['dissimilarity'] = 1
    #prediction_df['new_slide_prediction'] = new_slide_clf.predict(prediction_df[feature_names])
    #prediction_df.query("new_slide_prediction").to_csv(os.path.join(ocf_features_dif, video_file_name+"_new_slide.csv"), index=False)

    #return prediction_df 


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="data file with filename field")
    parser.add_argument("--img_sample_dir", help="data file with filename field", default=".")
    parser.add_argument("--max_duration", help="data file with filename field", default=float("inf"), type=int)
    args = parser.parse_args()
    
    
    main()