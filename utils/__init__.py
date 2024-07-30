from config import *
import os

if(not os.path.isdir(CHECKPOINT)):
    os.makedirs(CHECKPOINT)

if(not os.path.isdir(LOG_DIR)):
        os.makedirs(LOG_DIR)