from test import *
def read_image(file_path):
    with open(file_path, "rb") as file:
        image_bytes = file.read()
    return image_bytes
printable=read_image("trainingimage.jpg")
emotion_tell(printable)

