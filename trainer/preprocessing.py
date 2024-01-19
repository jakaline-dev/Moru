from datasets import Dataset
from PIL import Image
import os

def load_local_dataset_folder(folder_path):
    def gen():
        for image_path in os.listdir(folder_path):
            try:
                img = Image.open(image_path)
                base_name = os.path.splitext(os.path.basename(image_path))[:-1].join(".")
                text_path = os.path.join(folder_path, base_name + ".txt")
                yield {
                    "image": img,
                    "width": img.shape[0],
                    "height": img.shape[1],
                    "text": open(text_path, 'r').read() if open(text_path, 'r') else ''
                }
            except Exception:
                pass
    return Dataset.from_generator(gen)

if __name__ == "__main__":
    load_local_dataset_folder()