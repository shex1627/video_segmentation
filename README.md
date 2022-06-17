# video_segmentation

Given a tech talk video, this pipeline can find all the timestamp of all the new slides.  

#### Steps
1. setup virtual environment, install all the necessary packagese using requirements.txt  
2. run `python3 -m spacy download en_core_web_sm`   
3. install current package using `pip install -e .`  
4. use `block_skip_segmentation.py` or `main.py`. The former is faster. 
