import os
##################  VARIABLES  #####################
GCP_PROJECT = os.getenv("GCP_PROJECT")
BQ_DATASET = ""
BQ_REGION = "EU"
##################  CONSTANTS  #####################
LOCAL_PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
LOCAL_DATA_PATH = "/raw_data/ForestNetDataset"

# print(os.path.dirname(__file__))