from typing import Callable
import pandas as pd

class slideSegmentator:
    """class to find slide breakpoints from list of frames"""
    def __init__(self):
        return 

    def find_break_points(self):
        return

def dissimilaritySlideSegmentator(slideSegmentator):
    def __init__(self, func: Callable):
        self.compute_dist = func
    
    def create_dissimilar_df(self, img_dict: dict):
        total_pixals = img_dict[0].shape[0] * img_dict[0].shape[1]
        img_keys = sorted(img_dict)
        diff_dict = {}
        for index in range(len(img_keys)-1):
            dissimilarity = self.compute_dist(img_dict[img_keys[index+1]], img_dict[img_keys[index]])
            diff_dict[img_keys[index+1]] = {'dissimilarity': dissimilarity}
        diff_df = pd.DataFrame.from_dict(diff_dict, orient='index').reset_index()
        return diff_df


    def find_break_points(self, slide: pd.DataFrame):
        return
    