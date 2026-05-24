import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
from utils.file_utils import create_directory

class LegibilityClassifier:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self):
        """Load pretrained legibility model"""
        model = models.resnet34(pretrained=False)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 1)
        
        state_dict = torch.load(self.config.CLASSIFIER_WEIGHTS, map_location='cpu')
        new_state_dict = {k.replace("model_ft.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict)
        model.eval()
        self.model = model
        return self

    def filter_crops(self, crops_dir):
        """Classify and filter legible crops"""
        create_directory(self.config.LEGIBLE_CROPS_DIR)
        
        legible_paths = []
        for crop_file in os.listdir(crops_dir):
            if crop_file.endswith('.jpg'):
                crop_path = os.path.join(crops_dir, crop_file)
                if self._is_legible(crop_path):
                    legible_paths.append(crop_path)
                    self._move_legible(crop_path, crop_file)
        
        return legible_paths

    def _is_legible(self, image_path):
        """Classify single crop"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            return torch.sigmoid(output).item() > 0.5

    def _move_legible(self, src_path, filename):
        """Move legible crop to destination directory"""
        dest_path = os.path.join(self.config.LEGIBLE_CROPS_DIR, filename)
        os.rename(src_path, dest_path)
