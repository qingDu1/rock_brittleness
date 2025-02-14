import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class ImageFeatureExtractor:
    """Extract features from SEM images using pretrained CNN models"""

    def __init__(self, model_name='resnet50', use_gpu=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.model = self._load_model(model_name)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self, model_name):
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Remove the final fully connected layer
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model = model.to(self.device)
        model.eval()
        return model

    def extract(self, image_path):
        """Extract features from a single image"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            features = self.model(image)

        return features.cpu().numpy().flatten()


class EDSFeatureExtractor:
    """Process and extract features from EDS elemental analysis data"""

    def __init__(self, normalize=True):
        self.normalize = normalize
        self.element_list = ['O', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Fe']

    def extract(self, eds_data):
        """Extract features from EDS data"""
        # Extract basic elemental compositions
        compositions = np.array([eds_data[elem] for elem in self.element_list])

        if self.normalize:
            compositions = compositions / np.sum(compositions)

        # Calculate additional features
        features = {
            'compositions': compositions,
            'si_al_ratio': eds_data['Si'] / (eds_data['Al'] + 1e-6),
            'alkali_ratio': (eds_data['Na'] + eds_data['K']) / (eds_data['Ca'] + eds_data['Mg'] + 1e-6),
            'total_alkali': eds_data['Na'] + eds_data['K'] + eds_data['Ca'] + eds_data['Mg']
        }

        return features


class XRDFeatureExtractor:
    """Process and extract features from XRD mineral composition data"""

    def __init__(self, normalize=True):
        self.normalize = normalize

    def extract(self, xrd_data):
        """Extract features from XRD data"""
        # Extract mineral compositions
        mineral_contents = xrd_data.values

        if self.normalize:
            mineral_contents = mineral_contents / np.sum(mineral_contents)

        # Calculate mineralogical indices
        features = {
            'mineral_contents': mineral_contents,
            'quartz_content': xrd_data.get('Quartz', 0),
            'clay_content': sum(xrd_data.get(mineral, 0) for mineral in ['Illite', 'Kaolinite', 'Smectite']),
            'carbonate_content': sum(xrd_data.get(mineral, 0) for mineral in ['Calcite', 'Dolomite'])
        }

        return features


def combine_features(image_features, eds_features, xrd_features):
    """Combine features from different sources into a single feature vector"""
    combined = np.concatenate([
        image_features,
        eds_features['compositions'],
        [eds_features['si_al_ratio'],
         eds_features['alkali_ratio'],
         eds_features['total_alkali']],
        xrd_features['mineral_contents'],
        [xrd_features['quartz_content'],
         xrd_features['clay_content'],
         xrd_features['carbonate_content']]
    ])

    return combined