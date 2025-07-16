"Pormpt: give code of this question"
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import random
from skimage import measure
from skimage.measure import regionprops
import os
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.resnet import resnet34
from PIL import Image
from skimage import measure
from skimage.measure import regionprops
import cv2
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.resnet import resnet34
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Subset

# Define paths
train_image_dir = "./Datasets/Q2/detection/train/images"
train_mask_dir = "./Datasets/Q2/detection/train/masks"

# 3.1.1 Load the dataset and process masks to extract bounding box coordinates
def load_dataset(image_dir, mask_dir):
    """
    Load images and masks, and organize them into a dataset structure
    """
    dataset = []
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    for img_file in image_files:
        # Construct corresponding mask filename
        mask_file = img_file  # Same name as per the description
        
        # Check if mask file exists
        if not os.path.exists(os.path.join(mask_dir, mask_file)):
            print(f"Warning: Mask file {mask_file} not found for image {img_file}")
            continue
        
        # Add to dataset
        dataset.append({
            'image_path': os.path.join(image_dir, img_file),
            'mask_path': os.path.join(mask_dir, mask_file),
            'id': img_file.split('.')[0]
        })

    
    print(f"Loaded {len(dataset)} image-mask pairs")
    return dataset

dataset = load_dataset(train_image_dir, train_mask_dir)

# 3.1.2 Create a function to convert segmentation masks to bounding box annotations
def masks_to_boxes(mask):
    """
    Convert segmentation mask to bounding boxes
    
    Args:
        mask: numpy array of the mask image (H, W)
    
    Returns:
        boxes: numpy array of bounding boxes in format [x_min, y_min, x_max, y_max]
    """
    # Make sure mask is a numpy array
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    # Convert RGB to grayscale if needed
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    # Use connected components to find instances in the mask
    labeled_mask = measure.label(mask > 0)
    
    # Get regions properties
    regions = regionprops(labeled_mask)
    
    boxes = []
    for region in regions:
        # Get bounding box coordinates
        y_min, x_min, y_max, x_max = region.bbox
        
        # Add bounding box to the list
        # Ensure boxes are not degenerate (have positive width and height)
        if x_max > x_min and y_max > y_min:
            boxes.append([x_min, y_min, x_max, y_max])
    
    # Handle case when no regions are found
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    
    return np.array(boxes)



class FruitDetectionDataset(Dataset):
    def __init__(self, dataset, transform=None, augment=True):
        self.dataset = dataset
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        mask = np.array(Image.open(item["mask_path"]))

        # Extract bounding boxes
        boxes = masks_to_boxes(mask)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(boxes),), dtype=torch.int64)
        }
        if self.augment:
            image, target = self.apply_augmentation(image, target)

        if self.transform:
            image = self.transform(image)
        
        return image, target

    def apply_augmentation(self, image, target):
        image_tensor = F.to_tensor(image)
        # Random horizontal flip with 50% probability
        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            target["boxes"][:, [0, 2]] = image.width - target["boxes"][:, [2, 0]]

        # Color Jittering
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)
            image_tensor = F.adjust_brightness(image_tensor, brightness)
            image_tensor = F.adjust_contrast(image_tensor, contrast)
            image_tensor = F.adjust_saturation(image_tensor, saturation)
            image_tensor = F.adjust_hue(image_tensor, hue)

        return image, target

# Define transformations
transform = T.Compose([T.ToTensor()])

# Load dataset
train_images = load_dataset(train_image_dir,train_mask_dir)
train_datasetf = FruitDetectionDataset(train_images, transform=transform)

# Dataset splitting (80% train, 20% validation)
train_size = int(0.8 * len(train_datasetf))
val_size = len(train_datasetf) - train_size
train_indices = list(range(train_size))
val_indices = list(range(train_size, len(train_datasetf)))

# Use Subset to create the train and validation datasets
train_dataset = Subset(train_datasetf, train_indices)
val_dataset = Subset(train_datasetf, val_indices)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))



# Function to create Faster R-CNN with ResNet-34 backbone
def create_faster_rcnn():
    # Load pre-trained ResNet-34 and remove fully connected layers
    backbone = resnet34(pretrained=True)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  # Remove last FC layer
    backbone.out_channels = 512  # ResNet-34 feature maps output 512 channels

    # Define custom anchor sizes and aspect ratios for fruit detection
    anchor_sizes = ((16, 32, 64, 128, 256),)  # Multi-scale anchors
    aspect_ratios = ((0.5, 1.0, 2.0),)  # Standard aspect ratios

    # Create Anchor Generator
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # Create Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=2,  # Background + Fruit class
        rpn_anchor_generator=anchor_generator
    )

    return model
# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_faster_rcnn().to(device)


# Function to save model weights
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

# Function to load model weights
def load_model(model, save_path):
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        model.eval()  # Set the model to evaluation mode
        print(f"Model weights loaded from {save_path}")
    else:
        print("No pre-trained model found, starting training from scratch.")
model_save_path = 'fruit_q2.pth'
# Load model weights if they exist before training
load_model(model, model_save_path)


# Define optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())  # Sum all losses
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    lr_scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    save_model(model, model_save_path)

print("Model training complete")