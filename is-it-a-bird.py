from pathlib import Path

from ddgs import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
import warnings
import socket,warnings
import os
import time


try:
    socket.setdefaulttimeout(1)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('1.1.1.1', 53))
except socket.error as ex: raise Exception("STOP: No internet. Click '>|' in top right and set 'Internet' switch to on")

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')



warnings.filterwarnings("ignore")

def search_images(keywords, max_images=200): 
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')


for attempt in range(5):
    try:
        urls = search_images('bird photos', max_images=1)
        print(urls[0])
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(3) 


dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)


im = Image.open(dest)
im.to_thumb(256,256)


download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)


download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)

searches = 'forest','bird'
path = Path('bird_or_not')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo')) # type: ignore
    time.sleep(5)
    resize_images(path/o, max_size=400, dest=path/o) # type: ignore

failed = verify_images(get_image_files(path)) # type: ignore
failed.map(Path.unlink)
len(failed)

dls = DataBlock( # type: ignore
    blocks=(ImageBlock, CategoryBlock), # type: ignore
    get_items=get_image_files, # type: ignore
    splitter=RandomSplitter(valid_pct=0.2, seed=42), # type: ignore
    get_y=parent_label, # type: ignore
    item_tfms=[Resize(192, method='squish')] # type: ignore
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)

learn = vision_learner(dls, resnet18, metrics=error_rate) # type: ignore
learn.fine_tune(3)

is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))  # type: ignore
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")



# pip3 install ddgs
# pip3 install -q duckduckgo_search
# pip3 install fastdownload
# pip3 install fastai
# pip3 install fastcore
