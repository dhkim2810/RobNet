import os
import torch
import zipfile
import requests
from tqdm import tqdm

def download_weights():
    url = (
        "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
    )

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in Mebibyte
    total_size = int(r.headers.get("content-length", 0))
    block_size = 2 ** 20  # Mebibyte
    t = tqdm(total=total_size, unit="MiB", unit_scale=True)

    with open("state_dicts.zip", "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        raise Exception("Error, something went wrong")

    print("Download successful. Unzipping file...")
    path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
    directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
        print("Unzip file successful!")

def convert_weight(state_dict):
    chk = torch.load("cifar10_models/state_dicts/vgg16_bn.pt")
    convert = {
        'fc1.weight':'classifier.0.weight',
        'fc1.bias':'classifier.0.bias',
        'fc2.weight':'classifier.3.weight',
        'fc2.bias':'classifier.3.bias',
        'classifier.weight':'classifier.6.weight',
        'classifier.bias':'classifier.6.bias',
    }
    for name, m in state_dict.items():
        if name in convert.keys():
            state_dict[name] = chk[convert[name]]
        else:
            state_dict[name] = chk[name]
    
    return state_dict

def load_model(model):
    download_weights()
    new_state_dict = convert_weight(model.state_dict())
    model.load_state_dict(new_state_dict)
    return model