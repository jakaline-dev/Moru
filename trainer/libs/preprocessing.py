from PIL import Image

# Height, Width
SDXL_GRID = [
    # (512, 2048), # 4
    # (512, 1984),
    # (512, 1920),
    # (512, 1856),
    # (576, 1792),
    (576, 1728),  # 3
    (576, 1664),
    (640, 1600),
    (640, 1536),
    (704, 1472),
    (704, 1408),  # 2
    (704, 1344),
    (768, 1344),
    (768, 1280),
    (832, 1216),
    (832, 1152),
    (896, 1152),
    (896, 1088),
    (960, 1088),
    (960, 1024),
    (1024, 1024),  # 1
    (1024, 960),
    (1088, 960),
    (1088, 896),
    (1152, 896),
    (1152, 832),
    (1216, 832),
    (1280, 768),
    (1344, 768),
    (1344, 704),
    (1408, 704),  # 2
    (1472, 704),
    (1536, 640),
    (1600, 640),
    (1664, 576),
    (1728, 576),  # 3
    # (1792, 576),
    # (1856, 512),
    # (1920, 512),
    # (1984, 512),
    # (2048, 512) # 4
]
SDXL_GRID_RATIO = [h / w for (h, w) in SDXL_GRID]


def find_closest_index(sorted_list, x):
    return min(range(len(sorted_list)), key=lambda i: abs(sorted_list[i] - x))


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def get_target_size(image: Image):
    image_ratio = image.height / image.width
    target_size_idx = find_closest_index(SDXL_GRID_RATIO, image_ratio)
    target_size = SDXL_GRID[target_size_idx]
    target_ratio = SDXL_GRID_RATIO[target_size_idx]
    if target_ratio <= image_ratio:
        k = target_size[1] / image.width
        # resize_resolution = (int(k * image.height), target_size[1])
        resize_resolution = (target_size[1], int(k * image.height))
    else:
        k = target_size[0] / image.height
        # resize_resolution = (target_size[0], int(k * image.width))
        resize_resolution = (int(k * image.width), target_size[0])
    return target_size, resize_resolution


def fit_grid(example, downcast_original_sizes=True):
    image = example["image"]
    target_size, resize_res = get_target_size(image)
    example["original_sizes"] = tuple([image.height, image.width])
    if (
        image.height > resize_res[0]
        and image.width > resize_res[1]
        and downcast_original_sizes
    ):
        example["original_sizes"] = resize_res
    example["target_sizes"] = tuple(target_size)
    example["image"] = image.convert("RGB").resize(
        (resize_res[1], resize_res[0]), Image.Resampling.BICUBIC
    )
    return example
