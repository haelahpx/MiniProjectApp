import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.current_image = None
        self.overlay_image = None
        
    def load_image(self, image_path):
        self.original_image = cv2.imread(image_path)
        self.current_image = self.original_image.copy()
        return self.current_image
    
    def load_overlay_image(self, image_path):
        self.overlay_image = cv2.imread(image_path)
        return self.overlay_image
    
    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            return self.current_image
        return None
    
    def to_grayscale(self, weights=(0.299, 0.587, 0.114)):
        if self.current_image is None:
            return None
        return cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
    
    def negative_transform(self):
        if self.current_image is None:
            return None
        return cv2.bitwise_not(self.current_image)
    
    def apply_sepia(self):
        if self.current_image is None:
            return None
        img_sepia = np.array(self.current_image, dtype=np.float64)
        img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                       [0.349, 0.686, 0.168],
                                                       [0.393, 0.769, 0.189]]))
        img_sepia[np.where(img_sepia > 255)] = 255
        return np.array(img_sepia, dtype=np.uint8)
    
    def apply_cyanotype(self):
        if self.current_image is None:
            return None
        img_cyan = np.array(self.current_image, dtype=np.float64)
        img_cyan = cv2.transform(img_cyan, np.matrix([[0.1, 0.4, 0.4],
                                                     [0.2, 0.7, 0.7],
                                                     [0.4, 0.8, 0.8]]))
        img_cyan[np.where(img_cyan > 255)] = 255
        return np.array(img_cyan, dtype=np.uint8)
    
    def adjust_brightness_contrast(self, brightness=0, contrast=1):
        if self.current_image is None:
            return None
        adjusted = cv2.convertScaleAbs(self.current_image, alpha=contrast, beta=brightness)
        return adjusted
    
    def adjust_color_channels(self, red=1.0, green=1.0, blue=1.0):
        if self.current_image is None:
            return None
        b, g, r = cv2.split(self.current_image)
        r = cv2.multiply(r, red)
        g = cv2.multiply(g, green)
        b = cv2.multiply(b, blue)
        return cv2.merge([b, g, r])
    
    def color_filter(self, lower_range, upper_range):
        if self.current_image is None:
            return None
        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower_range), np.array(upper_range))
        return cv2.bitwise_and(self.current_image, self.current_image, mask=mask)
    
    def overlay_images(self, alpha=0.5, x_offset=0, y_offset=0):
        if self.current_image is None or self.overlay_image is None:
            return None
            
        overlay_resized = cv2.resize(self.overlay_image, 
                                   (self.current_image.shape[1], self.current_image.shape[0]))
        
        output = self.current_image.copy()
        cv2.addWeighted(overlay_resized, alpha, output, 1 - alpha, 0, output)
        return output
    
    def flip(self, direction):
        if self.current_image is None:
            return None
        if direction == 'horizontal':
            return cv2.flip(self.current_image, 1)
        elif direction == 'vertical':
            return cv2.flip(self.current_image, 0)
        elif direction == 'diagonal':
            return cv2.flip(cv2.flip(self.current_image, 1), 0)
        return self.current_image
    
    def rotate(self, angle):
        if self.current_image is None:
            return None
        rows, cols = self.current_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(self.current_image, M, (cols, rows))
    
    def scale(self, scale_factor):
        if self.current_image is None:
            return None
        width = int(self.current_image.shape[1] * scale_factor)
        height = int(self.current_image.shape[0] * scale_factor)
        return cv2.resize(self.current_image, (width, height))
    
    def apply_transform(self, rotation=0, scale=1.0, tx=0, ty=0):
        if self.current_image is None:
            return None
        rows, cols = self.current_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        return cv2.warpAffine(self.current_image, M, (cols, rows))
    
    def apply_fourier_transform(self, highpass=False):
        if self.current_image is None:
            return None
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        if highpass:
            rows, cols = gray.shape
            crow, ccol = rows//2, cols//2
            fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
            
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        return cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def apply_spatial_filter(self, filter_type='mean', kernel_size=3):
        if self.current_image is None:
            return None
            
        if filter_type == 'mean':
            return cv2.blur(self.current_image, (kernel_size, kernel_size))
        elif filter_type == 'gaussian':
            return cv2.GaussianBlur(self.current_image, (kernel_size, kernel_size), 0)
        elif filter_type == 'median':
            return cv2.medianBlur(self.current_image, kernel_size)
        return None
    
    def apply_edge_detection(self, method='sobel', threshold1=100, threshold2=200):
        if self.current_image is None:
            return None
            
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        if method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.magnitude(sobelx, sobely).astype(np.uint8)
        elif method == 'laplacian':
            return cv2.Laplacian(gray, cv2.CV_64F).astype(np.uint8)
        elif method == 'canny':
            return cv2.Canny(gray, threshold1, threshold2)
        return None
    
    def apply_histogram_equalization(self):
        if self.current_image is None:
            return None
        img_yuv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    def apply_contrast_stretching(self):
        if self.current_image is None:
            return None
        norm_image = cv2.normalize(self.current_image, None, 0, 255, cv2.NORM_MINMAX)
        return norm_image
    
    def apply_gamma_correction(self, gamma=1.0):
        if self.current_image is None:
            return None
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(self.current_image, table)
    
    def apply_morphological_operation(self, operation, kernel_size=3):
        if self.current_image is None:
            return None
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        if operation == 'dilate':
            return cv2.dilate(binary, kernel, iterations=1)
        elif operation == 'erode':
            return cv2.erode(binary, kernel, iterations=1)
        elif operation == 'opening':
            return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif operation == 'boundary':
            erosion = cv2.erode(binary, kernel, iterations=1)
            return binary - erosion
        elif operation == 'skeleton':
            return cv2.ximgproc.thinning(binary)
        return None

    def apply_segmentation(self, method='kmeans', n_segments=3):
        if self.current_image is None:
            return None
            
        if method == 'kmeans':
            pixel_values = self.current_image.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixel_values, n_segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            return segmented_data.reshape(self.current_image.shape)
        
        elif method == 'watershed':
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(self.current_image, markers)
            self.current_image[markers == -1] = [0, 0, 255]
            return self.current_image
