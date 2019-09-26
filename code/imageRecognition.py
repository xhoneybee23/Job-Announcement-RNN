
# coding: utf-8

# In[ ]:



# coding: utf-8

import io
import os
import re
from google.cloud import vision
from google.cloud.vision import types


def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    
    print('Texts:')
    
    
    print(type(texts))
    
    count = 0

    for text in texts:
        if count >= 2: break
        if text == None : count = count+1
        pharse = re.sub(r'[-=.,#/?:*$■※·@→I~←!+、o(")ㅇ○●㈜/\>①②③④}ㅣ|]','',format(text.description))
        print(pharse)

path = "image/1.jpg"
detect_text(path)


