import torch

from torch.nn.utils.rnn import pad_sequence


def pad_input_dict(dict_list, padding_value=0):
    """
    Takes a list of tokenized input dicts and returns a single dict
    with padded, stacked tensors.
    """
    collated = {}
    for key in dict_list[0].keys():
        sequences = [d[key] for d in dict_list]
        collated[key] = pad_sequence(sequences, batch_first=True, padding_value=padding_value).reshape(-1, 512)
    return collated

def collate_fn(batch):
    # Stack images
    images = {"pixel_values": torch.stack([b[0]["pixel_values"] for b in batch], dim=0).squeeze(1)} # [B, 3, H, W]

    # Pad text fields
    caption_inputs = pad_input_dict([b[1] for b in batch])
    text_inputs = pad_input_dict([b[2] for b in batch])

    return images, caption_inputs, text_inputs

def send_to_device(inputs, device):
    return {k: v.to(device) for k, v in inputs.items()}