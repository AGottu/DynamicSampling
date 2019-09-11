import os.path
import gdown

output1 = 'data/all_datasets/train/train.drop.jsonl'
if not os.path.isfile(output1):
    url1 = f'https://drive.google.com/uc?id=1fJcEE_7yV4oUUocyBdkLpmBu_Zkd60iY'    
    gdown.download(url1, output1, quiet=False)