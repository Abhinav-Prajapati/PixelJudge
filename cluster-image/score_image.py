import numpy as np
import cv2
from skimage.measure import shannon_entropy
from skimage.feature import local_binary_pattern
from scipy.stats import kurtosis, skew
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ImageQualityAnalyzer:
    def __init__(self):
        # Initialize pre-trained model for feature extraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Identity()  # Use the model as a feature extractor
        self.model.to(self.device)
        self.model.eval()
        
        # Transform for the model
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def analyze_image(self, image_path):
        """Analyze image quality attributes without using metadata."""
        # Load image in different formats for different analyses
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Convert to PIL Image for torchvision transforms
        pil_img = Image.fromarray(img_rgb)
        
        # Dictionary to store all results
        results = {
            "dimensions": img_bgr.shape[:2],  # (height, width)
            "aspect_ratio": img_bgr.shape[1] / img_bgr.shape[0]
        }
        
        # Add detailed analyses
        results.update(self._analyze_noise(img_gray))
        results.update(self._analyze_sharpness(img_gray))
        results.update(self._analyze_brightness_contrast(img_gray, img_hsv))
        results.update(self._analyze_color_distribution(img_rgb, img_hsv))
        results.update(self._analyze_texture(img_gray))
        results.update(self._estimate_iso_equivalent(img_gray))
        results.update(self._deep_feature_analysis(pil_img))
        
        return results
    
    def _analyze_noise(self, img_gray):
        """Estimate noise level in the image."""
        # Apply median filter to remove noise
        denoised = cv2.medianBlur(img_gray, 5)
        
        # Calculate noise as difference between original and denoised
        noise = img_gray.astype(np.float32) - denoised.astype(np.float32)
        
        # Calculate noise metrics
        noise_std = np.std(noise)
        noise_energy = np.mean(noise**2)
        
        # Apply wavelet decomposition for noise estimation
        noise_wavelet = self._wavelet_noise_estimation(img_gray)
        
        return {
            "noise_level": noise_std,
            "noise_energy": noise_energy,
            "noise_wavelet_estimate": noise_wavelet
        }
    
    def _wavelet_noise_estimation(self, img_gray):
        """Estimate noise using wavelet decomposition."""
        # Simple implementation using Laplacian
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        return np.std(laplacian) / np.sqrt(2)
    
    def _analyze_sharpness(self, img_gray):
        """Analyze image sharpness."""
        # Laplacian for edge detection (higher values indicate more sharp edges)
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        sharpness_laplacian = np.var(laplacian)
        
        # Sobel operators for gradient magnitude
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sharpness_gradient = np.mean(gradient_magnitude)
        
        # Use FFT to analyze frequency components
        f_transform = np.fft.fft2(img_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # High frequency energy ratio indicates sharpness
        h, w = img_gray.shape
        center_y, center_x = h // 2, w // 2
        mask_radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask_area = (y - center_y)**2 + (x - center_x)**2 <= mask_radius**2
        
        total_energy = np.sum(magnitude_spectrum)
        low_freq_energy = np.sum(magnitude_spectrum * mask_area)
        high_freq_energy = total_energy - low_freq_energy
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        return {
            "sharpness_laplacian": sharpness_laplacian,
            "sharpness_gradient": sharpness_gradient,
            "high_frequency_content": high_freq_ratio
        }
    
    def _analyze_brightness_contrast(self, img_gray, img_hsv):
        """Analyze image brightness and contrast."""
        # Brightness metrics
        mean_brightness = np.mean(img_gray)
        median_brightness = np.median(img_gray)
        
        # Contrast metrics
        contrast_std = np.std(img_gray)
        contrast_range = np.max(img_gray) - np.min(img_gray)
        
        # Histogram analysis
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Exposure analysis from HSV
        v_channel = img_hsv[:,:,2]
        underexposed_ratio = np.sum(v_channel < 50) / v_channel.size
        overexposed_ratio = np.sum(v_channel > 205) / v_channel.size
        
        return {
            "mean_brightness": mean_brightness,
            "median_brightness": median_brightness,
            "contrast_std": contrast_std,
            "contrast_range": contrast_range,
            "underexposed_ratio": underexposed_ratio,
            "overexposed_ratio": overexposed_ratio
        }
    
    def _analyze_color_distribution(self, img_rgb, img_hsv):
        """Analyze color distribution and saturation."""
        # Calculate mean and std for each RGB channel
        r_mean, g_mean, b_mean = np.mean(img_rgb[:,:,0]), np.mean(img_rgb[:,:,1]), np.mean(img_rgb[:,:,2])
        r_std, g_std, b_std = np.std(img_rgb[:,:,0]), np.std(img_rgb[:,:,1]), np.std(img_rgb[:,:,2])
        
        # Calculate mean and std for saturation
        saturation = img_hsv[:,:,1]
        mean_saturation = np.mean(saturation)
        std_saturation = np.std(saturation)
        
        # Color distribution metrics
        dominant_hue_hist = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(dominant_hue_hist)
        
        # Color balance - difference between RGB channels
        color_balance_rg = r_mean - g_mean
        color_balance_rb = r_mean - b_mean
        color_balance_gb = g_mean - b_mean
        
        return {
            "color_r_mean": r_mean,
            "color_g_mean": g_mean,
            "color_b_mean": b_mean,
            "color_r_std": r_std,
            "color_g_std": g_std,
            "color_b_std": b_std,
            "mean_saturation": mean_saturation,
            "std_saturation": std_saturation,
            "dominant_hue": dominant_hue,
            "color_balance_rg": color_balance_rg,
            "color_balance_rb": color_balance_rb,
            "color_balance_gb": color_balance_gb
        }
    
    def _analyze_texture(self, img_gray):
        """Analyze image texture."""
        # Local Binary Pattern for texture analysis
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        
        # Calculate texture metrics
        lbp_entropy = shannon_entropy(lbp)
        
        # Gray-Level Co-occurrence Matrix (simplified)
        gray_scaled = ((img_gray / 16).astype(np.uint8) * 16)  # Quantize to 16 levels
        
        # Calculate statistical properties
        img_entropy = shannon_entropy(img_gray)
        img_kurtosis = kurtosis(img_gray.flatten())
        img_skewness = skew(img_gray.flatten())
        
        return {
            "texture_lbp_entropy": lbp_entropy,
            "texture_entropy": img_entropy,
            "texture_kurtosis": img_kurtosis,
            "texture_skewness": img_skewness
        }
    
    def _estimate_iso_equivalent(self, img_gray):
        """Estimate ISO equivalent from noise characteristics."""
        # First, analyze noise in uniform areas
        blocks = self._extract_uniform_blocks(img_gray)
        
        if not blocks:
            # If no uniform areas found, use whole image noise
            noise_std = np.std(img_gray - cv2.GaussianBlur(img_gray, (5, 5), 0))
        else:
            # Calculate noise in uniform areas
            noise_levels = []
            for block in blocks:
                block_noise = np.std(block - cv2.GaussianBlur(block, (5, 5), 0))
                noise_levels.append(block_noise)
            noise_std = np.mean(noise_levels)
        
        # Apply a simple model to map noise to ISO
        # This is a very simplified model and would need calibration
        # ISO ~ k * noise_std^2 (for actual cameras)
        k = 100  # This constant would need calibration with known images
        estimated_iso = k * (noise_std ** 2)
        
        # Clamp to reasonable ISO values
        estimated_iso = max(50, min(25600, estimated_iso))
        
        return {
            "estimated_iso": estimated_iso,
            "noise_std_in_flat_areas": noise_std
        }
    
    def _extract_uniform_blocks(self, img_gray, block_size=16, threshold=10):
        """Extract uniform blocks for noise analysis."""
        h, w = img_gray.shape
        blocks = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = img_gray[y:y+block_size, x:x+block_size]
                if np.std(block) < threshold:  # If block is uniform
                    blocks.append(block)
        
        return blocks
    
    def _deep_feature_analysis(self, pil_img):
        """Use deep learning for feature extraction and analysis."""
        # Transform image for the model
        img_tensor = self.transform(pil_img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # Use features for quality prediction
        # This is simplified - normally you'd use a trained classifier on these features
        features_np = features.cpu().numpy()[0]
        
        # Calculate basic statistics of deep features
        feature_mean = np.mean(features_np)
        feature_std = np.std(features_np)
        feature_min = np.min(features_np)
        feature_max = np.max(features_np)
        
        # Use PCA to reduce dimensions (simplified)
        # For a real implementation, consider using sklearn's PCA
        feature_pca = np.mean(features_np.reshape(32, -1), axis=1)
        
        return {
            "deep_feature_mean": feature_mean,
            "deep_feature_std": feature_std,
            "deep_feature_range": feature_max - feature_min,
            "deep_feature_pca": feature_pca.tolist()[:5]  # First 5 components
        }

def predict_quality_score(analyzer_results, weights=None):
    """
    Convert analyzer results to an overall quality score.
    This is a simplified model - a real implementation would use ML.
    """
    # Default weights for different aspects of quality
    if weights is None:
        weights = {
            "noise": -0.3,
            "sharpness": 0.25,
            "contrast": 0.15,
            "exposure": -0.2,
            "color": 0.1
        }
    
    # Extract relevant metrics
    noise_score = 1.0 / (1.0 + analyzer_results["noise_level"])
    sharpness_score = min(1.0, analyzer_results["sharpness_gradient"] / 50)
    contrast_score = min(1.0, analyzer_results["contrast_std"] / 80)
    
    # Penalize under/overexposure
    exposure_penalty = (analyzer_results["underexposed_ratio"] + 
                        analyzer_results["overexposed_ratio"]) * 2
    exposure_score = 1.0 - exposure_penalty
    
    # Color score based on saturation
    color_score = min(1.0, analyzer_results["mean_saturation"] / 128)
    
    # Combine scores
    total_score = (
        weights["noise"] * noise_score +
        weights["sharpness"] * sharpness_score +
        weights["contrast"] * contrast_score +
        weights["exposure"] * exposure_score +
        weights["color"] * color_score
    )
    
    # Normalize to 0-100 scale
    normalized_score = max(0, min(100, (total_score + 0.5) * 100))
    
    return normalized_score

def main():
    # Example usage
    analyzer = ImageQualityAnalyzer()
    
    image_path = "/home/abhinav/workspace/openai-clip/media/1.jpg"  # Replace with actual path
    
    try:
        results = analyzer.analyze_image(image_path)
        
        # Calculate overall quality score
        quality_score = predict_quality_score(results)
        results["overall_quality_score"] = quality_score
        
        # Print results in a readable format
        print(f"Image Quality Analysis Results for: {image_path}")
        print(f"Dimensions: {results['dimensions'][1]}x{results['dimensions'][0]}")
        print(f"Aspect Ratio: {results['aspect_ratio']:.2f}")
        print(f"Estimated ISO: {results['estimated_iso']:.0f}")
        print(f"Noise Level: {results['noise_level']:.2f}")
        print(f"Sharpness: {results['sharpness_gradient']:.2f}")
        print(f"Mean Brightness: {results['mean_brightness']:.2f}/255")
        print(f"Contrast: {results['contrast_std']:.2f}")
        print(f"Mean Saturation: {results['mean_saturation']:.2f}/255")
        print(f"Overexposed Areas: {results['overexposed_ratio']*100:.1f}%")
        print(f"Underexposed Areas: {results['underexposed_ratio']*100:.1f}%")
        print(f"Overall Quality Score: {results['overall_quality_score']:.1f}/100")
        
    except Exception as e:
        print(f"Error analyzing image: {e}")

if __name__ == "__main__":
    main()
