import cv2
import numpy as np
import os
import random

class Augmentor:
    def __init__(self):
        self.augment_count = 10  # Number of augmented versions per image

    def augment_image(self, image_path, output_folder, augmented_mode):
        """
        Applies augmentation transformations if augmented mode is enabled.
        The original image is named with '_aug0', and augmented images are '_aug1', '_aug2', etc.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return []

        # Ensure output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")

        augmented_images = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        if augmented_mode:
            original_aug_path = os.path.join(output_folder, f"{base_name}_aug0.jpg")
            success = cv2.imwrite(original_aug_path, img)
            if success:
                print(f"Saved original image: {original_aug_path}")
            else:
                print(f"Error saving original image: {original_aug_path}")
            augmented_images.append(original_aug_path)

            for i in range(1, self.augment_count + 1):
                aug_img = self.apply_random_transformation(img)
                aug_path = os.path.join(output_folder, f"{base_name}_aug{i}.jpg")
                success = cv2.imwrite(aug_path, aug_img)
                if success:
                    print(f"Saved augmented image: {aug_path}")
                else:
                    print(f"Error saving augmented image: {aug_path}")
                augmented_images.append(aug_path)
        else:
            original_path = os.path.join(output_folder, f"{base_name}.jpg")
            success = cv2.imwrite(original_path, img)
            if success:
                print(f"Saved original image (no augmentation): {original_path}")
            else:
                print(f"Error saving original image: {original_path}")
            augmented_images.append(original_path)

        return augmented_images
    def apply_random_transformation(self, img):
        """
        Applies a random augmentation to the given image.
        """
        transformations = [
            self.rotate, self.add_noise, self.shift_perspective, self.adjust_hue,
            self.adjust_brightness_contrast, self.gaussian_blur, self.random_crop_resize, self.affine_transform
        ]
        return random.choice(transformations)(img)

    def rotate(self, img):
        """ Rotates the image within a limited range. """
        angle = random.uniform(-25, 25)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    def add_noise(self, img):
        """ Adds random noise to the image. """
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        return cv2.add(img, noise)

    def shift_perspective(self, img):
        """ Applies a random perspective shift. """
        h, w = img.shape[:2]
        pts1 = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
        shift = random.uniform(-20, 20)
        pts2 = np.float32([[shift, shift], [w-shift, shift], [shift, h-shift], [w-shift, h-shift]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h))

    def adjust_hue(self, img):
        """ Adjusts the hue of the image randomly. """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def adjust_brightness_contrast(self, img):
        """ Randomly changes brightness and contrast. """
        alpha = random.uniform(0.7, 1.3)  # Contrast
        beta = random.randint(-30, 30)  # Brightness
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def gaussian_blur(self, img):
        """ Applies Gaussian blur to the image. """
        ksize = random.choice([3, 5])  # Kernel size
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    def random_crop_resize(self, img):
        """ Crops a portion of the image and resizes it back. """
        h, w = img.shape[:2]
        crop_x = random.randint(5, w//6)
        crop_y = random.randint(5, h//6)
        cropped = img[crop_y:h-crop_y, crop_x:w-crop_x]
        return cv2.resize(cropped, (w, h))

    def affine_transform(self, img):
        """ Applies affine transformation. """
        h, w = img.shape[:2]
        src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1]])
        dst_pts = np.float32([[random.randint(-10, 10), random.randint(-10, 10)], 
                              [w-1+random.randint(-10, 10), random.randint(-10, 10)], 
                              [random.randint(-10, 10), h-1+random.randint(-10, 10)]])
        M = cv2.getAffineTransform(src_pts, dst_pts)
        return cv2.warpAffine(img, M, (w, h))
