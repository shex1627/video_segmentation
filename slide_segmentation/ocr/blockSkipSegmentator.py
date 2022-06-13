from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict
from slide_segmentation.util import *
from slide_segmentation.ocr.ocr_process import *
from slide_segmentation.ocr.config import *
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import logging 
from dataclasses import dataclass, field


logger = logging.getLogger("skip_segmentation")

class BlockSkipSegmentator:
    def __init__(self, chunk_size: int, img_file_dict: dict, 
                        ocr_model: ..., clf_model: ..., 
                        doc_diff_comparator: docDiffBuilder=doc_diff_comparator) -> None:
        self.chunk_size = chunk_size
        self.ocr = ocr_model
        self.doc_diff_comparator = doc_diff_comparator
        self.clf_model = clf_model
        self.img_data_dict = defaultdict(lambda : None)
        self.img_data_dict.update(img_file_dict)
        self.ocr_data_dict = defaultdict(lambda : None)


    def get_ocr_result(self, frame_index:str):
        if frame_index in self.ocr_data_dict:
            return self.ocr_data_dict[frame_index]
        img_data = self.img_data_dict[frame_index]
        if img_data.shape[0] > 1:
            ocr_result = self.ocr.ocr(img_data)
            self.ocr_data_dict[frame_index] = ocr_result
            return ocr_result
        else:
            return None
        
    def _generate_ocr_doc_feature(self, ocr_result1, ocr_result2, doc_diff_comparator: docDiffBuilder=doc_diff_comparator):
        """
        generate feature feature based on ocr results
        """
        doc1 = get_paragraph(ocr_result1)
        doc2 = get_paragraph(ocr_result2)
        doc_compare_dict = doc_diff_comparator.compare(doc1, doc2)
        doc_compare_dict['frame_token_ct'] = len(doc1)
        # need to test if dataframe results results
        feature_df = pd.DataFrame([doc_compare_dict])
        feature_df = feature_df.rename(columns={'letter_dis':'letter_dissim'})
        return feature_df


    def check_if_frame_break_point(self, frame_index):
        """
        check if the frame is a breakpoint compared to previous frame
        """
        if frame_index == 0:
            return {'new_slide_prediction':False, 'index': frame_index}
        #curr_frame, prev_frame = img_data_dict[frame_index], img_data_dict[frame_index-1]
        return self.check_if_frame_diff(frame_index,frame_index-1)


    def check_if_frame_diff(self, frame_1_index, frame_2_index):
        """
        check if two frames are different
        """
        empty_result = {'new_slide_prediction':False, 'index': frame_1_index}
        ocr1, ocr2 = self.get_ocr_result(frame_1_index), self.get_ocr_result(frame_2_index)
        try:
            feature_df = self._generate_ocr_doc_feature(ocr1, ocr2)
            if_break_point = self.clf_model.predict(feature_df)[0]
            feature_df['index'] = frame_1_index
            feature_df['new_slide_prediction'] = if_break_point
        except Exception as e:
            logger.info(e, exc_info=True)
            return empty_result
        return feature_df.to_dict('records')[0]


    def find_all_new_slides(self, start_frame_index: int, end_frame_index: int):
        """find all the new slides given frame index interval and the sklearn model."""
        new_slide_indices = []
        for i in range(start_frame_index+1, end_frame_index+1):
            is_new_slide_record = self.check_if_frame_break_point(i)
            if is_new_slide_record['new_slide_prediction']:
                new_slide_indices.append(is_new_slide_record)
        return new_slide_indices


    def get_slide_breakpoints(self, max_block_count: int=float("inf")):
        img_files_blocks = np.array_split(list(self.img_data_dict), len(self.img_data_dict)//self.chunk_size)
        break_points = []

        i = 0
        for img_files_block in img_files_blocks:
            logger.info("#####")
            logger.info("looking at new block")
            logger.info("#####")
            # check if the first one is split point
            frame_index = img_files_block[0]
            break_point_dict = self.check_if_frame_break_point(frame_index)
            is_break_point = break_point_dict['new_slide_prediction']
            if is_break_point:
                break_points.append(break_point_dict)
            last_file_frame_index = img_files_block[-1]
            
            compare_head_tail_result = self.check_if_frame_diff(frame_index, last_file_frame_index)
            same_slide_block = not compare_head_tail_result['new_slide_prediction']
            logger.info(frame_index)
            logger.info(is_break_point)
            logger.info(f"{frame_index} is {is_break_point} breakpoint")
            logger.info(f"block index range: {frame_index, last_file_frame_index}")
            logger.info(f"same slide block: {same_slide_block}")
            logger.info("\n\n")
            
            block_new_slides = []
            if not same_slide_block:
                block_new_slides = self.find_all_new_slides(frame_index, last_file_frame_index)
            logger.info(f"block_new_slides: {block_new_slides}")
            break_points += block_new_slides
            i+=1
            if i == max_block_count:
                break
        return pd.DataFrame(break_points)



