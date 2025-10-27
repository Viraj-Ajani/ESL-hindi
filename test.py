import os
import sys
from lib import evaluation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## for heuristic_strategy, note that you should set the flag varibale 'heuristic_strategy'  in line 44 of ves.py as True
# RUN_PATH = "../checkpoint_heuristic_flickr30k_bert.tar"


## for adaptive_strategy, note that you should set the flag varibale 'heuristic_strategy'  in line 44 of ves.py as False
# RUN_PATH = "./checkpoint2/model_best.pth"


DATA_PATH = "./Flickr30K/"
evaluation.evalrank(sys.argv[1], sys.argv[-1], data_path=DATA_PATH, split="test")
