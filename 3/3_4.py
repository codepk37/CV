import os
import clip
import torch
from PIL import Image
import requests
from tqdm import tqdm
import torchvision.transforms as T
from torchvision import models
import torch.nn.functional as F
import shutil

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)

# Load ImageNet class names
def load_imagenet_classes(file_path="imagenet_classes.txt",
                          url="https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"):
    if not os.path.exists(file_path):
        print("Downloading ImageNet labels...")
        response = requests.get(url)
        with open(file_path, "w") as f:
            f.write(response.text)
    with open(file_path, "r") as f:
        return f.read().strip().split("\n")

imagenet_classes = load_imagenet_classes()
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in imagenet_classes]).to(device)

# Label mapping
label_dict = {
    "n01498041": "stingray",
    "n01582220": "magpie",
    "n01667114": "mud turtle",
    "n01687978": "agama",
    "n01749939": "green mamba",
    "n01751748": "sea snake",
    "n01818515": "macaw",
    "n01824575": "coucal",
    "n01883070": "wombat",
    "n02007558": "flamingo"
}

# Normalize text features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=1, keepdim=True)

# Load ResNet50 (ImageNet pretrained)
imagenet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
imagenet_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def show_top5_imagenet(image_tensor):
    with torch.no_grad():
        outputs = imagenet_model(image_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        top5 = torch.topk(probs, 5)
        return [imagenet_classes[i] for i in top5.indices], [probs[i].item()*100 for i in top5.indices]

# Make results folder
results_dir = "Results3_4"
os.makedirs(results_dir, exist_ok=True)

# Main loop
root_dir = "/media/pavan/STORAGE/linux_storage/CV_all/assignment-5-codepk37/3/data"

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    GT_label = label_dict.get(folder)
    if not GT_label:
        continue

    clip_correct_only = 0
    resnet_correct_only = 0

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        try:
            # Open and preprocess image for CLIP
            image = Image.open(image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)

            # CLIP predictions
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                logits_per_image = image_features @ text_features.t() * logit_scale
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            top5_clip_indices = probs[0].argsort()[-5:][::-1]
            top5_clip_labels = [imagenet_classes[i] for i in top5_clip_indices]

            clip_result = GT_label in top5_clip_labels

            # ResNet predictions
            imagenet_input = imagenet_transform(image).unsqueeze(0).to(device)
            labels, scores = show_top5_imagenet(imagenet_input)
            resnet_result = GT_label in labels

            # Check logic
            if clip_result and not resnet_result and clip_correct_only < 2:
                # Save this image as CLIP success only
                new_name = os.path.splitext(filename)[0] + "_clip.jpg"
                save_path = os.path.join(results_dir, new_name)
                image.save(save_path)
                clip_correct_only += 1

            elif resnet_result and not clip_result and resnet_correct_only < 1:
                # Save this image as ResNet success only
                new_name = os.path.splitext(filename)[0] + "_resnet.jpg"
                save_path = os.path.join(results_dir, new_name)
                image.save(save_path)
                resnet_correct_only += 1

            # Stop if enough images collected for this folder
            if clip_correct_only >= 2 and resnet_correct_only >= 1:
                print(f"Done with folder {folder}")
                break

        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
