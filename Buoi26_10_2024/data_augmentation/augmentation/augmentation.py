import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json

class YOLODataAugmentation:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        self.brightness = config.get("brightness", 0.2)
        self.contrast = config.get("contrast", 0.2)
        self.saturation = config.get("saturation", 0.2)
        self.scale = config.get("scale", 1.2)
        self.gaussian_blur_kernel_size = config.get("gaussian_blur_kernel_size", 5)
        self.motion_blur_size = config.get("motion_blur_size", 5)
        self.noise_mean = config.get("noise_mean", 0)
        self.noise_sigma = config.get("noise_sigma", 25)
        self.crop_x = config["random_crop"]["crop_x"]
        self.crop_y = config["random_crop"]["crop_y"]
        self.crop_w = config["random_crop"]["crop_w"]
        self.crop_h = config["random_crop"]["crop_h"]

    def horizontal_flip(self, image: np.ndarray)->np.ndarray:
        """Flip the image horizontally, while adjusting the bouding box coordinates accordingly."""    
        image = cv2.flip(image, 1)
        return image
    
    def vertical_flip(self, image:  np.ndarray)->np.ndarray:
        """Flip the image vertically, while adjusting the bouding box coordinates accordingly."""
        image = cv2.flip(image, 0)
        return image
    
    def rotate_image(self, image:  np.ndarray, angle: float)->np.ndarray:
        """Rotate the image by a specific angle and update the bounding boxes based on the rotation angle."""
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))
        return rotated_image
    
    def random_scale(self, image:  np.ndarray, scale=1.2)->np.ndarray:
        """Change the image size while keeping the bouding boxes proportional."""
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        scaled_image = cv2.resize(image, (new_w, new_h))
        return scaled_image
    
    def random_crop(self, image:  np.ndarray, crop_x: int, crop_y: int, crop_w: int, crop_h: int)->np.ndarray:
        """Randomly crop a region of the image, ensuring bounding boxes remain within the new region."""
        h, w = image.shape[:2]
        crop_w = min(crop_w, w - crop_x)
        crop_h = min(crop_h, h - crop_y)

        cropped_image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        return cropped_image
    
    def color_jitter(self, image: np.ndarray, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2)->np.ndarray:
        """Adjust the birghtness, contrast, saturation, and color of the image by modifying its color channels."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_image[..., 1] *= 1 + random.uniform(-saturation, saturation)
        hsv_image[..., 2] *= 1 + random.uniform(-brightness, brightness)
        image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype(np.float32)
        yuv_image[..., 0] *= 1 + random.uniform(-contrast, contrast) 
        yuv_image[..., 0] = np.clip(yuv_image[..., 0], 0, 255) 
        image = cv2.cvtColor(yuv_image.astype(np.uint8), cv2.COLOR_YUV2BGR)
        return image
    
    def gaussian_blur(self, image: np.ndarray, kernal_size: int = 5)->np.ndarray:
        return cv2.GaussianBlur(image, (kernal_size, kernal_size), 0)
    
    def motion_blur(self, image: np.ndarray, size: int = 5)->np.ndarray:
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        return cv2.filter2D(image, -1, kernel_motion_blur)
    
    def add_noise(self, image: np.ndarray, mean: int = 0, sigma: int = 25)->np.ndarray:
        """Add Gaussian noise to the image to improve the model's robustness."""
        gauss = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, gauss)
        return noisy_image
    

image = cv2.imread(r"D:\Machine Learning HIT14\AI_2024\Buoi26_10_2024\image\meo.jpeg")
config_path = r'D:\Machine Learning HIT14\AI_2024\Buoi26_10_2024\config\config.json'
augmentor = YOLODataAugmentation(config_path)
augmentations = {
    "Original": image,
    "Horizontal Flip": augmentor.horizontal_flip(image),
    "Vertical Flip": augmentor.vertical_flip(image),
    "Rotate 30°": augmentor.rotate_image(image, angle=30),
    "Random Scale": augmentor.random_scale(image),
    "Random Crop": augmentor.random_crop(image, crop_x=50, crop_y=50, crop_w=200, crop_h=200),
    "Color Jitter": augmentor.color_jitter(image),
    "Gaussian Blur": augmentor.gaussian_blur(image),
    "Motion Blur": augmentor.motion_blur(image),
    "Add Noise": augmentor.add_noise(image) 
}

output_folder = r"D:\Machine Learning HIT14\AI_2024\Buoi26_10_2024\output"
os.makedirs(output_folder, exist_ok=True)

for name, aug_img in augmentations.items():
    file_path = os.path.join(output_folder, f"{name.replace(' ', '_').replace('°', '')}.jpeg")  # Đổi tên file để dễ đọc
    cv2.imwrite(file_path, aug_img)

