#!/usr/bin/env python3
"""
Image Grouping Script - Groups images based on content similarity and capture time
"""

import os
import glob
from datetime import datetime
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import torch
from torchvision import models, transforms
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

class ImageGrouper:
    def __init__(self, image_folder, output_folder='grouped_images', 
                 time_weight=0.3, eps=0.5, min_samples=3):
        """
        Initialize the ImageGrouper object
        
        Parameters:
        -----------
        image_folder : str
            Path to folder containing images
        output_folder : str
            Path where grouped images will be saved
        time_weight : float
            Weight of time features in clustering (0 to 1)
        eps : float
            DBSCAN epsilon parameter (maximum distance between samples)
        min_samples : int
            DBSCAN min_samples parameter (minimum samples in a cluster)
        """
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.time_weight = time_weight
        self.eps = eps
        self.min_samples = min_samples
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Setup model for feature extraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """Load pretrained ResNet model and prepare it for feature extraction"""
        print("Loading pretrained model...")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final fully connected layer
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.to(self.device)
        model.eval()
        return model
    
    def _extract_datetime(self, image_path):
        """Extract the date and time when the image was taken"""
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'DateTimeOriginal':
                        # Convert to timestamp
                        dt = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                        return dt.timestamp()
            
            # Fallback to file creation time if EXIF data not available
            return os.path.getctime(image_path)
        except Exception as e:
            print(f"Error extracting datetime from {image_path}: {e}")
            return os.path.getctime(image_path)
    
    def _extract_features(self, image_path):
        """Extract features from image using pretrained model"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_t)
            
            # Flatten the features
            return features.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def process_images(self):
        """Process all images and group them based on content and time"""
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.image_folder, ext)))
            image_files.extend(glob.glob(os.path.join(self.image_folder, ext.upper())))
        
        if not image_files:
            print(f"No images found in {self.image_folder}")
            return
        
        print(f"Processing {len(image_files)} images...")
        
        # Extract features and timestamps
        features = []
        timestamps = []
        valid_paths = []
        
        for img_path in tqdm(image_files, desc="Extracting image features"):
            img_features = self._extract_features(img_path)
            if img_features is not None:
                features.append(img_features)
                timestamps.append(self._extract_datetime(img_path))
                valid_paths.append(img_path)
        
        if not features:
            print("No valid features extracted. Exiting.")
            return
        
        # Convert lists to numpy arrays
        features = np.array(features)
        timestamps = np.array(timestamps).reshape(-1, 1)
        
        # Normalize features and timestamps
        scaler_features = StandardScaler()
        normalized_features = scaler_features.fit_transform(features)
        
        scaler_time = StandardScaler()
        normalized_time = scaler_time.fit_transform(timestamps)
        
        # Combine features and time with appropriate weighting
        content_weight = 1 - self.time_weight
        combined_features = np.concatenate([
            normalized_features * content_weight,
            normalized_time * self.time_weight
        ], axis=1)
        
        # Cluster the images
        print("Clustering images...")
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean').fit(combined_features)
        labels = clustering.labels_
        
        # Process the clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Found {n_clusters} clusters")
        
        # Group images by cluster
        for i, label in enumerate(labels):
            if label == -1:
                # Handle noise points (no cluster)
                cluster_dir = os.path.join(self.output_folder, "unclustered")
            else:
                cluster_dir = os.path.join(self.output_folder, f"cluster_{label}")
            
            if not os.path.exists(cluster_dir):
                os.makedirs(cluster_dir)
            
            # Copy image to appropriate cluster directory
            img_path = valid_paths[i]
            dest_path = os.path.join(cluster_dir, os.path.basename(img_path))
            shutil.copy2(img_path, dest_path)
        
        print(f"Images have been grouped into {self.output_folder}")
        return labels, valid_paths
    
    def visualize_clusters(self, labels, image_paths, samples_per_cluster=3):
        """Visualize sample images from each cluster"""
        if not labels.any():
            print("No labels to visualize")
            return
            
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points
                
            # Get indices of images in this cluster
            indices = np.where(labels == label)[0]
            if len(indices) == 0:
                continue
                
            # Select sample images
            sample_indices = indices[:min(samples_per_cluster, len(indices))]
            
            # Create a figure for this cluster
            fig, axes = plt.subplots(1, len(sample_indices), figsize=(4*len(sample_indices), 4))
            if len(sample_indices) == 1:
                axes = [axes]
                
            fig.suptitle(f"Cluster {label} - {len(indices)} images")
            
            # Display sample images
            for i, idx in enumerate(sample_indices):
                img_path = image_paths[idx]
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(os.path.basename(img_path))
                axes[i].axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_folder, f"cluster_{label}_samples.png"))
            plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Group images based on content and time')
    parser.add_argument('--input', type=str, required=True, help='Input folder containing images')
    parser.add_argument('--output', type=str, default='grouped_images', help='Output folder for grouped images')
    parser.add_argument('--time-weight', type=float, default=0.3, 
                        help='Weight for time information (0-1). Higher means time is more important.')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN epsilon parameter')
    parser.add_argument('--min-samples', type=int, default=3, help='DBSCAN min_samples parameter')
    parser.add_argument('--visualize', action='store_true', help='Visualize clusters')
    
    args = parser.parse_args()
    
    grouper = ImageGrouper(
        image_folder=args.input,
        output_folder=args.output,
        time_weight=args.time_weight,
        eps=args.eps,
        min_samples=args.min_samples
    )
    
    labels, paths = grouper.process_images()
    
    if args.visualize and labels is not None:
        grouper.visualize_clusters(labels, paths)


if __name__ == "__main__":
    main()
