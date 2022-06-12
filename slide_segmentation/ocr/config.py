from slide_segmentation.util import *
from slide_segmentation.ocr.ocr_process import *
from joblib import dump, load

MODEL_PATH = 'C:\\Users\\alistar\\Desktop\\ds\\video_segmentation\\models\\ocr_tree.joblib'
feature_names = ['jaccard', 'jaccard_letter', 'frame_token_ct',
       'word_dis', 'letter_dissim'] #'dissimilarity', 
new_slide_clf = load(MODEL_PATH)    

doc_diff_comparator = docDiffBuilder().\
        add_metric("jaccard_letter", jaccard_letter_dissim).\
        add_metric("jaccard", jaccard_str).\
        add_metric("letter_dis",letter_dissim).\
        add_metric("word_dis", word_dissim)