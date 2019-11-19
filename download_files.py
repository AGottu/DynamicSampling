import os.path
import gdown

output1 = 'data/all_datasets.zip'
if not os.path.isfile(output1):
    #url1 = f'https://drive.google.com/uc?id=1gSOd0N0QiaEv1E5ipp6zvzXVdmgcdjvn' # New Ropes
    url1 = f'https://drive.google.com/uc?id=18TaondJzLPEEFqov32EK1A0fDk39bFWE' # Old Ropes
    #url1 = f'https://drive.google.com/uc?id=1MnJe7aGbCVW8lLi5vjlw-hQNp2FopVwp' # Old Ropes Train, New Ropes Dev
    
    
    #url1 = f'https://drive.google.com/uc?id=1Cq0ibIUqwYCL1Z77gVrzGXWXPx4172JY'
    #url1 = f'https://drive.google.com/uc?id=1QYfG0mV75RXzi5RkHGedZGItWvJUwRnG'
    gdown.download(url1, output1, quiet=False)

output2 = 'bert/pytorch_model.bin'
if not os.path.isfile(output2):
    url2 = f'https://drive.google.com/uc?id=1ljo1d5ZQcfAMZBAHhyfNjtkqT2UmzAeB' 
    #gdown.download(url2, output2, quiet=False)