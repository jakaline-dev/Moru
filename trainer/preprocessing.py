from datasets import Dataset, Value
from PIL import Image
import os


def gen(folder_path: str):
    #folder_path = r"D:\Dataset\jdkd\1_jdkdface"
    for filename in os.listdir(folder_path):
        try:
            image_path = os.path.join(folder_path, filename)
            if not os.path.isfile(image_path) or os.path.splitext(filename)[1] == ".txt":
                continue
            img = Image.open(image_path)
            base_name = os.path.splitext(filename)[0]
            text_path = os.path.join(folder_path, base_name + ".txt")
            if not os.path.isfile(text_path):
                continue
            text = open(text_path, 'r').read().strip()
            item = {
                "image": img,
                "width": img.size[0],
                "height": img.size[1],
                "text": text
            }
            yield item
        except Exception as e:
            print(e)
            pass

def load_local_dataset_folder(folder_path):
    #, features={"image": Image.Image, "width": Value(dtype="int32"), "height": Value(dtype="int32"), "text": Value(dtype="string")}
    return Dataset.from_generator(gen, gen_kwargs={"folder_path": folder_path})