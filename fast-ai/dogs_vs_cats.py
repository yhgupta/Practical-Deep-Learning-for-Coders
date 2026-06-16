from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')

def predict(img):
    _, _, probs = learn.predict(PILImage.create(img))
    return {"Dog": float(probs[0]), "Cat": float(probs[1])}


image = gr.Image(type='pil')
label = gr.Label()
examples = ['dog.jpg', 'cat.jpg', 'dunno.jpg']

intf = gr.Interface(fn=predict, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)