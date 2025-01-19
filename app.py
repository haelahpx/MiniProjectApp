import flet as ft
import cv2
import numpy as np
from PIL import Image as PILImage
import base64
from io import BytesIO
from enum import Enum
import math
import os
from datetime import datetime

class ProcessingMode(Enum):
    BASIC = "Basic Operations"
    MATHEMATICAL = "Mathematical Operations"
    TRANSFORMS = "Transforms & Filters"
    ENHANCEMENT = "Image Enhancement"
    COMPRESSION = "Image Compression"
    SEGMENTATION = "Image Segmentation"
    BINARY = "Binary Operations"
    RESTORATION = "Image Restoration"
    MATCHING = "Image Matching"

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

def main(page: ft.Page):
    page.title = "Advanced Image Processing Application"
    page.padding = 20
    page.scroll = "adaptive"
    
    processor = ImageProcessor()
    current_image_data = None  # Store the current image data for downloading
    
    image_container = ft.Container(
        content=None,
        alignment=ft.alignment.center,
        bgcolor=ft.colors.BLACK12,
        border_radius=10,
        padding=10,
        expand=True,
        height=500,
    )

    def update_image_display(cv_image):
        if cv_image is None:
            return
            
        if len(cv_image.shape) == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        elif len(cv_image.shape) == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
        pil_image = PILImage.fromarray(cv_image)
        
        aspect_ratio = pil_image.width / pil_image.height
        max_width = min(800, page.window_width * 0.6)
        max_height = min(600, page.window_height * 0.7)
        
        if aspect_ratio > max_width / max_height:
            width = max_width
            height = width / aspect_ratio
        else:
            height = max_height
            width = height * aspect_ratio
            
        pil_image.thumbnail((int(width), int(height)), PILImage.Resampling.LANCZOS)
        
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        global current_image_data
        current_image_data = buffered.getvalue()  # Store for download
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        image_container.content = ft.Image(
            src_base64=img_base64,
            fit=ft.ImageFit.CONTAIN,
            border_radius=10,
        )
        page.update()

    def download_image(e):
        if current_image_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_image_{timestamp}.png"
            
            # Save the image to the downloads directory
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
            filepath = os.path.join(downloads_path, filename)
            
            with open(filepath, "wb") as f:
                f.write(current_image_data)
            
            page.show_snack_bar(ft.SnackBar(
                content=ft.Text(f"Image saved to Downloads folder as {filename}")
            ))

    def handle_file_picker_result(e: ft.FilePickerResultEvent):
        if not e.files:
            return
        
        file_path = e.files[0].path
        try:
            cv_image = processor.load_image(file_path)
            update_image_display(cv_image)
            tools_column.visible = True
            download_button.disabled = False  # Enable download button
            page.update()
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error loading image: {str(ex)}")))

    def handle_overlay_picker_result(e: ft.FilePickerResultEvent):
        if not e.files:
            return
        
        file_path = e.files[0].path
        try:
            processor.load_overlay_image(file_path)
            update_image_display(processor.overlay_images(float(overlay_opacity_slider.value)))
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error loading overlay: {str(ex)}")))

    def handle_spatial_filter(e):
        filter_type = e.control.text.lower().split()[0]
        kernel_size = int(spatial_kernel_slider.value)
        if kernel_size % 2 == 0:
            kernel_size += 1
        result = processor.apply_spatial_filter(filter_type, kernel_size)
        update_image_display(result)

    def handle_edge_detection(e):
        method = e.control.text.lower()
        threshold1 = float(edge_threshold1_slider.value)
        threshold2 = float(edge_threshold2_slider.value)
        result = processor.apply_edge_detection(method, threshold1, threshold2)
        update_image_display(result)

    def handle_morphological_operation(e):
        operation = e.control.text.lower()
        kernel_size = int(morph_kernel_slider.value)
        result = processor.apply_morphological_operation(operation, kernel_size)
        update_image_display(result)

    def reset_image(e):
        try:
            result = processor.reset_image()
            update_image_display(result)
            # Reset all sliders to default values
            brightness_slider.value = 0
            contrast_slider.value = 1
            blur_slider.value = 1
            edge_threshold1_slider.value = 100
            edge_threshold2_slider.value = 200
            red_slider.value = 1
            green_slider.value = 1
            blue_slider.value = 1
            rotation_slider.value = 0
            scale_slider.value = 1
            kmeans_slider.value = 3
            page.update()   
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error resetting image: {str(ex)}")))

    def reset_sliders():
        overlay_opacity_slider.value = 0.5
        spatial_kernel_slider.value = 3
        edge_threshold1_slider.value = 100
        edge_threshold2_slider.value = 200
        morph_kernel_slider.value = 3
        gamma_slider.value = 1.0
        brightness_slider.value = 0
        contrast_slider.value = 1.0
        red_slider.value = 1.0
        green_slider.value = 1.0
        blue_slider.value = 1.0
        rotation_slider.value = 0
        scale_slider.value = 1.0
        segments_slider.value = 3

    # Real-time update handlers for sliders
    def on_brightness_contrast_change(e):
        result = processor.adjust_brightness_contrast(
            brightness=brightness_slider.value,
            contrast=contrast_slider.value
        )
        update_image_display(result)

    def on_color_channels_change(e):
        result = processor.adjust_color_channels(
            red=red_slider.value,
            green=green_slider.value,
            blue=blue_slider.value
        )
        update_image_display(result)

    def on_transform_change(e):
        result = processor.apply_transform(
            rotation=rotation_slider.value,
            scale=scale_slider.value
        )
        update_image_display(result)

    def on_overlay_change(e):
        result = processor.overlay_images(float(overlay_opacity_slider.value))
        update_image_display(result)

    def on_gamma_change(e):
        result = processor.apply_gamma_correction(float(gamma_slider.value))
        update_image_display(result)

    def apply_border_padding():
    # Ambil nilai slider dan warna dari input
        border_thickness = border_slider.value
        padding = padding_slider.value
        border_color = border_color_field.value if border_color_field.value else "#000000"
        padding_color = padding_color_field.value if padding_color_field.value else "#FFFFFF"

        try:
            # Terapkan border dan padding dengan warna ke container
            image_container.border = ft.border.all(border_thickness, border_color)
            image_container.bgcolor = padding_color
            image_container.padding = padding
            page.update()
        except Exception as ex:
            print(f"Error applying border and padding: {ex}")

    # Create file pickers
    file_picker = ft.FilePicker(on_result=handle_file_picker_result)
    overlay_picker = ft.FilePicker(on_result=handle_overlay_picker_result)
    page.overlay.extend([file_picker, overlay_picker])

    # Create sliders with real-time updates
    brightness_slider = ft.Slider(
        min=-100, max=100, value=0, label="Brightness",
        on_change=on_brightness_contrast_change
    )
    contrast_slider = ft.Slider(
        min=0.1, max=3.0, value=1.0, label="Contrast",
        on_change=on_brightness_contrast_change
    )
    red_slider = ft.Slider(
        min=0, max=2.0, value=1.0, label="Red",
        on_change=on_color_channels_change
    )
    green_slider = ft.Slider(
        min=0, max=2.0, value=1.0, label="Green",
        on_change=on_color_channels_change
    )
    blue_slider = ft.Slider(
        min=0, max=2.0, value=1.0, label="Blue",
        on_change=on_color_channels_change
    )
    rotation_slider = ft.Slider(
        min=-180, max=180, value=0, label="Rotation",
        on_change=on_transform_change
    )
    scale_slider = ft.Slider(
        min=0.1, max=2.0, value=1.0, label="Scale",
        on_change=on_transform_change
    )
    gamma_slider = ft.Slider(
        min=0.1, max=3.0, value=1.0, label="Gamma",
        on_change=on_gamma_change
    )
    overlay_opacity_slider = ft.Slider(
        min=0, max=1, value=0.5, label="Overlay Opacity",
        on_change=on_overlay_change
    )
    border_slider = ft.Slider(
        min=0, max=50, value=1, label="Border Thickness",
        on_change=lambda e: apply_border_padding()
    )
    padding_slider = ft.Slider(
        min=0, max=50, value=1, label="Padding",
        on_change=lambda e: apply_border_padding()
    )
    border_color_field = ft.TextField(
        label="Border Color (Hex)", 
        hint_text="e.g., #FF5733",
        on_change=lambda e: apply_border_padding()
    )
    padding_color_field = ft.TextField(
        label="Padding Color (Hex)", 
        hint_text="e.g., #FFF",
        on_change=lambda e: apply_border_padding()
    )
    spatial_kernel_slider = ft.Slider(min=3, max=15, value=3, label="Kernel Size")
    edge_threshold1_slider = ft.Slider(min=0, max=255, value=100, label="Threshold 1")
    edge_threshold2_slider = ft.Slider(min=0, max=255, value=200, label="Threshold 2")
    morph_kernel_slider = ft.Slider(min=3, max=15, value=3, label="Kernel Size")
    segments_slider = ft.Slider(min=2, max=8, value=3, label="Segments")

    # Download button
    download_button = ft.ElevatedButton(
        "Download Image",
        icon=ft.icons.DOWNLOAD,
        on_click=download_image,
        disabled=True
    )

    # Create panels
    basic_panel = ft.Column([
        ft.Text("Basic Effects", size=16, weight=ft.FontWeight.BOLD),
        ft.ElevatedButton("Grayscale", 
                         on_click=lambda _: update_image_display(processor.to_grayscale())),
        ft.ElevatedButton("Negative", 
                         on_click=lambda _: update_image_display(processor.negative_transform())),
    ], spacing=10)

    brightness_contrast_panel = ft.Column([
        ft.Text("Brightness & Contrast", size=16, weight=ft.FontWeight.BOLD),
        brightness_slider,
        contrast_slider,
    ], spacing=10)

    color_adjustment_panel = ft.Column([
        ft.Text("Color Adjustment", size=16, weight=ft.FontWeight.BOLD),
        red_slider,
        green_slider,
        blue_slider,
    ], spacing=10)

    transform_panel = ft.Column([
        ft.Text("Rotation", size=16, weight=ft.FontWeight.BOLD),
        rotation_slider,
        ft.Text("Transforms", size=16, weight=ft.FontWeight.BOLD),
        scale_slider,
        ft.Text("Flip", size=16, weight=ft.FontWeight.BOLD),
        ft.Row([
            ft.IconButton(
                icon=ft.icons.FLIP, 
                on_click=lambda _: update_image_display(processor.flip('horizontal'))
            ),
            ft.IconButton(
                icon=ft.icons.FLIP_CAMERA_ANDROID,
                on_click=lambda _: update_image_display(processor.flip('vertical'))
            ),
            ft.ElevatedButton("Diagonal Flip",
                on_click=lambda _: update_image_display(processor.flip('diagonal'))
            ),
        ], spacing=10),
    ], spacing=10)

    border_padding_panel = ft.Column([
        ft.Text("Border and Padding", size=16, weight=ft.FontWeight.BOLD),
        border_slider,
        border_color_field,
        padding_slider,
        padding_color_field,
    ], spacing=10)


    color_effects_panel = ft.Column([
        ft.Text("Color Effects", size=16, weight=ft.FontWeight.BOLD),
        ft.ElevatedButton("Sepia", 
                         on_click=lambda _: update_image_display(processor.apply_sepia())),
        ft.ElevatedButton("Cyanotype", 
                         on_click=lambda _: update_image_display(processor.apply_cyanotype())),
    ], spacing=10)

    overlay_panel = ft.Column([
        ft.Text("Image Overlay", size=16, weight=ft.FontWeight.BOLD),
        ft.ElevatedButton("Add Overlay Image", 
                         on_click=lambda _: overlay_picker.pick_files(
                             allowed_extensions=["png", "jpg", "jpeg", "bmp"]
                         )),
        overlay_opacity_slider,
    ], spacing=10)

    filters_panel = ft.Column([
        ft.Text("Spatial Filters", size=16, weight=ft.FontWeight.BOLD),
        spatial_kernel_slider,
        ft.Row([
            ft.ElevatedButton("Mean Filter", on_click=handle_spatial_filter),
            ft.ElevatedButton("Gaussian Filter", on_click=handle_spatial_filter),
            ft.ElevatedButton("Median Filter", on_click=handle_spatial_filter),
        ], spacing=10, wrap=True),
    ], spacing=10)

    edge_panel = ft.Column([
        ft.Text("Edge Detection", size=16, weight=ft.FontWeight.BOLD),
        edge_threshold1_slider,
        edge_threshold2_slider,
        ft.Row([
            ft.ElevatedButton("Sobel", on_click=handle_edge_detection),
            ft.ElevatedButton("Laplacian", on_click=handle_edge_detection),
            ft.ElevatedButton("Canny", on_click=handle_edge_detection),
        ], spacing=10, wrap=True),
    ], spacing=10)

    enhancement_panel = ft.Column([
        ft.Text("Enhancement", size=16, weight=ft.FontWeight.BOLD),
        ft.ElevatedButton("Histogram Equalization", 
                         on_click=lambda _: update_image_display(
                             processor.apply_histogram_equalization()
                         )),
        ft.ElevatedButton("Contrast Stretching", 
                         on_click=lambda _: update_image_display(
                             processor.apply_contrast_stretching()
                         )),
        gamma_slider,
    ], spacing=10)

    segmentation_panel = ft.Column([
        ft.Text("Segmentation", size=16, weight=ft.FontWeight.BOLD),
        segments_slider,
        ft.Row([
            ft.ElevatedButton("K-means", 
                            on_click=lambda _: update_image_display(
                                processor.apply_segmentation(
                                    method='kmeans',
                                    n_segments=int(segments_slider.value)
                                ))),
            ft.ElevatedButton("Watershed",
                            on_click=lambda _: update_image_display(
                                processor.apply_segmentation(method='watershed'))),
        ], spacing=10),
    ], spacing=10)

    binary_panel = ft.Column([
        ft.Text("Binary Operations", size=16, weight=ft.FontWeight.BOLD),
        morph_kernel_slider,
        ft.Row([
            ft.ElevatedButton("Dilate", on_click=handle_morphological_operation),
            ft.ElevatedButton("Erode", on_click=handle_morphological_operation),
        ], spacing=10, wrap=True),
        ft.Row([
            ft.ElevatedButton("Opening", on_click=handle_morphological_operation),
            ft.ElevatedButton("Closing", on_click=handle_morphological_operation),
        ], spacing=10, wrap=True),
        ft.Row([
            ft.ElevatedButton("Boundary", on_click=handle_morphological_operation),
            ft.ElevatedButton("Skeleton", on_click=handle_morphological_operation),
        ], spacing=10, wrap=True),
    ], spacing=10)

    # Create scrollable tools column
    tools_column = ft.Column(
        [
            ft.Row([
                ft.ElevatedButton("Reset Image", on_click=reset_image),
                download_button,
            ], spacing=10),
            basic_panel,
            brightness_contrast_panel,
            color_adjustment_panel,
            transform_panel,
            color_effects_panel,
            overlay_panel,
            filters_panel,
            edge_panel,
            enhancement_panel,
            segmentation_panel,
            binary_panel,
            border_padding_panel,
        ],
        spacing=20,
        visible=False,
        scroll=ft.ScrollMode.AUTO,
        height=page.window_height,
    )

    # Create responsive layout
    def create_responsive_layout():
        tools_width = min(300, page.window_width * 0.25)
        content_width = page.window_width - tools_width - 40
        
        tools_column.width = tools_width
        tools_column.height = page.window_height
        
        image_container.width = content_width
        
        page.update()

    def on_resize(e):
        create_responsive_layout()
        
    page.on_resize = on_resize

    # Create main layout
    main_row = ft.Row(
        [
            ft.Container(
                content=tools_column,
                border=ft.border.all(1, ft.colors.GREY_400),
                border_radius=10,
                padding=10
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("Preview", size=16, weight=ft.FontWeight.BOLD),
                    image_container
                ]),
                expand=True
            )
        ],
        spacing=20,
        alignment=ft.MainAxisAlignment.START,
    )

    page.add(
        ft.Column([
            ft.Container(
                content=ft.Column([
                    ft.Text("Advanced Image Processing", size=24, weight=ft.FontWeight.BOLD),
                    ft.ElevatedButton(
                        "Upload Image",
                        icon=ft.icons.UPLOAD_FILE,
                        on_click=lambda _: file_picker.pick_files(
                            allowed_extensions=["png", "jpg", "jpeg", "bmp"]
                        )
                    ),
                ], spacing=10),
                padding=ft.padding.only(bottom=20),
            ),
            main_row,
        ])
    )

    # Add keyboard shortcuts
    def on_keyboard(e: ft.KeyboardEvent):
        if e.key == "R" and e.ctrl:
            reset_image(None)

    page.on_keyboard_event = on_keyboard

    # Initial layout setup
    create_responsive_layout()

if __name__ == "__main__":
    ft.app(target=main)