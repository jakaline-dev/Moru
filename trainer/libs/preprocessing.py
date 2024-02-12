import os

import torch
from PIL import Image
from tqdm.auto import tqdm

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


def load_from_local(folder_path, downcast_original_sizes=True):
    print("Load local folder")
    data = []
    for filename in tqdm(os.listdir(folder_path)):
        try:
            image_path = os.path.join(folder_path, filename)
            if (
                not os.path.isfile(image_path)
                or os.path.splitext(filename)[1] == ".txt"
            ):
                continue
            image = Image.open(image_path)
            # pre-resize
            target_sizes, resize_res = get_target_size(image)
            if not (image.height == resize_res[0] and image.width == resize_res[1]):
                image = image.resize(resize_res, Image.Resampling.BICUBIC)
            # if image.mode == "RGBA":
            #     # No transparency allowed in PNGs - change alpha to white
            #     white_background = Image.new("RGB", image.size, (255, 255, 255))
            #     white_background.paste(image, (0, 0), mask=image.split()[3])
            #     image = white_background
            image = image.convert("RGB")

            base_name = os.path.splitext(filename)[0]
            text_path = os.path.join(folder_path, base_name + ".txt")
            if not os.path.isfile(text_path):
                continue
            text = open(text_path, "r").read().strip().splitlines()
            if len(text) == 1:
                text = text[0]

            if (
                image.height > resize_res[0]
                and image.width > resize_res[1]
                and downcast_original_sizes
            ):
                original_sizes = tuple([resize_res[1], resize_res[0]])
            else:
                original_sizes = tuple([image.height, image.width])

            data.append(
                {
                    "image": image,
                    "text": text,
                    "target_sizes": target_sizes,
                    "original_sizes": original_sizes,
                }
            )
        except Exception as e:
            print(e)
    return data


def find_closest_index(sorted_list, x):
    return min(range(len(sorted_list)), key=lambda i: abs(sorted_list[i] - x))


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


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt(text_encoders, input_ids_prompt, input_ids_pooled=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if i == 0:
            prompt_embeds = text_encoder(
                torch.where(input_ids_prompt == 0, 49407, input_ids_prompt),
                output_hidden_states=True,
                return_dict=False,
            )
        else:
            prompt_embeds = text_encoder(
                input_ids_prompt, output_hidden_states=True, return_dict=False
            )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def split_input_ids_chunks(t):
    if len(t) <= 75:
        return t
    if len(t) > 225:
        t = t[:225]
    # , == 267
    indices = (t == 267).nonzero(as_tuple=True)[0]
    if len(indices) == 0:
        # no tags
        if len(t) <= 150:
            return [t[:75], t[75:]]
        else:
            return [t[:75], t[75:150], t[150:]]
    # split by ,
    split_1 = indices[indices <= 75].max()
    if len(t[split_1 + 1 :]) <= 75:
        return [t[:split_1], t[split_1 + 1 :]]
    else:
        split_2 = indices[(split_1 <= indices) & (indices <= split_1 + 75)].max()
        return [t[: split_1 - 2], t[split_1 - 1 : split_2 - 1], t[split_2:]]


def pad_input_ids(tensor_list):
    longest_tensor = max(tensor_list, key=lambda t: t.size(1))
    chunks = -((longest_tensor.size(1) - 2) // -75)
    total_tensor = torch.zeros((len(tensor_list), 77 * chunks), dtype=torch.long)
    for idx, t in enumerate(tensor_list):
        t = split_input_ids_chunks(t[idx, 1:-1])
        if torch.is_tensor(t):
            total_tensor[idx, 1 : 1 + len(t)] = t
            total_tensor[idx, 0] = 49406
            total_tensor[idx, 1 + len(t)] = 49407
        else:
            for j, t_chunk in enumerate(t):
                total_tensor[idx, 1 + j * 77 : 1 + j * 77 + len(t_chunk)] = t_chunk
                total_tensor[idx, j * 77] = 49406
                total_tensor[idx, 1 + j * 77 + len(t_chunk)] = 49407
    return total_tensor


def encode_prompt(text_encoders, input_ids_prompt, input_ids_pooled=None):
    total_length = input_ids_prompt.shape[1]
    emb_prompt_concat = []
    emb_pooled_concat = []
    num_slices = (total_length + 77 - 1) // 77

    for i in range(num_slices):
        start = i * 77
        end = min(start + 77, total_length)
        emb_prompt, emb_pooled = _encode_prompt(
            text_encoders, input_ids_prompt[:, start:end]
        )
        emb_prompt_concat.append(emb_prompt)
        emb_pooled_concat.append(emb_pooled)
    emb_prompt = torch.cat(emb_prompt_concat, dim=1)
    # emb_pooled = torch.cat(emb_pooled_concat, dim=-1)[::1280*num_slices]
    emb_pooled = torch.mean(torch.stack(emb_pooled_concat), dim=0)
    return emb_prompt, emb_pooled
