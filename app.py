import flet as ft
import cv2
import numpy as np
from PIL import Image as PILImage
import base64
from io import BytesIO
from enum import Enum
import math

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
        
    def load_image(self, image_path):
        self.original_image = cv2.imread(image_path)
        self.current_image = self.original_image.copy()
        return self.current_image
    
    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            return self.current_image
        return None
    
    # Basic Operations
    def to_grayscale(self, weights=(0.299, 0.587, 0.114)):
        if self.current_image is None:
            return None
        return cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
    
    def negative(self):
        if self.current_image is None:
            return None
        return 255 - self.current_image
    
    def adjust_color(self, red=1.0, green=1.0, blue=1.0):
        if self.current_image is None:
            return None
        img = self.current_image.copy()
        img[:,:,0] = cv2.multiply(img[:,:,0], blue)  # OpenCV uses BGR
        img[:,:,1] = cv2.multiply(img[:,:,1], green)
        img[:,:,2] = cv2.multiply(img[:,:,2], red)
        return np.clip(img, 0, 255).astype(np.uint8)
    
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

    def translate(self, x_offset, y_offset):
        if self.current_image is None:
            return None
        rows, cols = self.current_image.shape[:2]
        M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        return cv2.warpAffine(self.current_image, M, (cols, rows))
    
    def scale(self, scale_factor):
        if self.current_image is None:
            return None
        width = int(self.current_image.shape[1] * scale_factor)
        height = int(self.current_image.shape[0] * scale_factor)
        dim = (width, height)
        return cv2.resize(self.current_image, dim, interpolation=cv2.INTER_AREA)
    
    def rotate(self, angle):
        if self.current_image is None:
            return None
        rows, cols = self.current_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(self.current_image, M, (cols, rows))
    
    def adjust_contrast(self, factor):
        if self.current_image is None:
            return None
        return cv2.convertScaleAbs(self.current_image, alpha=factor, beta=0)

    def apply_gaussian_blur(self, kernel_size):
        if self.current_image is None:
            return None
        # Ensure kernel size is odd
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(self.current_image, (kernel_size, kernel_size), 0)

    def apply_canny_edge(self, threshold1, threshold2):
        if self.current_image is None:
            return None
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, threshold1, threshold2)

    def apply_kmeans_segmentation(self, k):
        if self.current_image is None:
            return None
        vectorized = self.current_image.reshape((-1,3))
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        return segmented_data.reshape(self.current_image.shape)

def main(page: ft.Page):
    page.title = "Advanced Image Processing Application"
    page.padding = 20
    
    # Initialize image processor
    processor = ImageProcessor()

    # Image display container
    image_container = ft.Container(
        content=ft.Stack([]),  # Use a Stack to overlay the selection rectangle
        alignment=ft.alignment.center,
        bgcolor=ft.colors.BLACK12,
        border_radius=10,
        padding=10,
        width=800,
        height=600,
    )

    def update_image_display(cv_image):
        if cv_image is None:
            return
            
        # Convert to RGB for display
        if len(cv_image.shape) == 2:  # Grayscale
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        elif len(cv_image.shape) == 3:  # Color
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
        # Convert to PIL Image
        pil_image = PILImage.fromarray(cv_image)
        
        # Resize while maintaining aspect ratio
        pil_image.thumbnail((800, 600), PILImage.Resampling.LANCZOS)
        
        # Convert to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Update container
        image_container.content = ft.Image(
            src_base64=img_base64,
            fit=ft.ImageFit.CONTAIN,
            border_radius=10,
        )
        page.update()

    def handle_file_picker_result(e: ft.FilePickerResultEvent):
        if not e.files:
            return
            
        file_path = e.files[0].path
        try:
            cv_image = processor.load_image(file_path)
            update_image_display(cv_image)
            # Enable processing controls
            tools_column.visible = True
            page.update()
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error loading image: {str(ex)}")))

    # Slider change handlers
    def on_brightness_change(e):
        try:
            if processor.current_image is None:
                return  # Ensure there's an image to process

            # Adjust brightness directly
            value = float(e.control.value)
            hsv = cv2.cvtColor(processor.current_image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Update the image display
            update_image_display(result)

        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error adjusting brightness: {str(ex)}")))


    def on_contrast_change(e):
        try:
            result = processor.adjust_contrast(float(e.control.value))
            update_image_display(result)
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error adjusting contrast: {str(ex)}")))

    def on_blur_change(e):
        try:
            result = processor.apply_gaussian_blur(float(e.control.value))
            update_image_display(result)
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error applying blur: {str(ex)}")))

    def on_edge_threshold_change(e):
        try:
            threshold1 = float(edge_threshold1_slider.value)
            threshold2 = float(edge_threshold2_slider.value)
            result = processor.apply_canny_edge(threshold1, threshold2)
            update_image_display(result)
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error detecting edges: {str(ex)}")))

    def on_color_change(e):
        try:
            red = float(red_slider.value)
            green = float(green_slider.value)
            blue = float(blue_slider.value)
            result = processor.adjust_color(red, green, blue)
            update_image_display(result)
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error adjusting colors: {str(ex)}")))

    def on_rotation_change(e):
        try:
            result = processor.rotate(float(e.control.value))
            update_image_display(result)
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error rotating image: {str(ex)}")))

    def on_scale_change(e):
        try:
            result = processor.scale(float(e.control.value))
            update_image_display(result)
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error scaling image: {str(ex)}")))

    def on_kmeans_change(e):
        try:
            result = processor.apply_kmeans_segmentation(int(e.control.value))
            update_image_display(result)
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error applying segmentation: {str(ex)}")))

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

    file_picker = ft.FilePicker(on_result=handle_file_picker_result)
    page.overlay.append(file_picker)

    # Create sliders
    brightness_slider = ft.Slider(min=-100, max=100, value=0, label="Brightness", on_change=on_brightness_change)
    contrast_slider = ft.Slider(min=0.1, max=3.0, value=1, label="Contrast", on_change=on_contrast_change)
    blur_slider = ft.Slider(min=1, max=99, value=1, label="Blur", on_change=on_blur_change)
    edge_threshold1_slider = ft.Slider(min=0, max=255, value=100, label="Edge Threshold 1", on_change=on_edge_threshold_change)
    edge_threshold2_slider = ft.Slider(min=0, max=255, value=200, label="Edge Threshold 2", on_change=on_edge_threshold_change)
    red_slider = ft.Slider(min=0, max=2, value=1, label="Red", on_change=on_color_change)
    green_slider = ft.Slider(min=0, max=2, value=1, label="Green", on_change=on_color_change)
    blue_slider = ft.Slider(min=0, max=2, value=1, label="Blue", on_change=on_color_change)
    rotation_slider = ft.Slider(min=-180, max=180, value=0, label="Rotation", on_change=on_rotation_change)
    scale_slider = ft.Slider(min=0.1, max=2.0, value=1, label="Scale", on_change=on_scale_change)
    kmeans_slider = ft.Slider(min=2, max=8, value=3, label="K-means Clusters", on_change=on_kmeans_change)

    # Create tool panels
    basic_adjustments = ft.Column([
        ft.Text("Basic Adjustments", size=16, weight=ft.FontWeight.BOLD),
        ft.Text("      Brightness", size=14, weight=ft.FontWeight.BOLD),
        brightness_slider,
        ft.Text("      Contrast", size=14, weight=ft.FontWeight.BOLD),
        contrast_slider,
        ft.Text("      Blur", size=14, weight=ft.FontWeight.BOLD),
        blur_slider,
    ], spacing=10)

    color_adjustments = ft.Column([
        ft.Text("Color Adjustments", size=16, weight=ft.FontWeight.BOLD),
        ft.Text("      Red", size=14, weight=ft.FontWeight.BOLD),
        red_slider,
        ft.Text("      Green", size=14, weight=ft.FontWeight.BOLD),
        green_slider,
        ft.Text("      Blue", size=14, weight=ft.FontWeight.BOLD),
        blue_slider,
    ], spacing=10)

    edge_detection = ft.Column([
        ft.Text("Edge Detection", size=16, weight=ft.FontWeight.BOLD),
        ft.Text("      Edge Threshold 1", size=14, weight=ft.FontWeight.BOLD),
        edge_threshold1_slider,
        ft.Text("      Edge Threshold 2", size=14, weight=ft.FontWeight.BOLD),
        edge_threshold2_slider,
    ], spacing=10)

    transform_controls = ft.Column([
        ft.Text("Transforms", size=16, weight=ft.FontWeight.BOLD),
        ft.Text("      Rotation", size=14, weight=ft.FontWeight.BOLD),
        rotation_slider,
        ft.Text("      Scale", size=14, weight=ft.FontWeight.BOLD),
        scale_slider,
    ], spacing=10)

    segmentation_controls = ft.Column([
        ft.Text("Segmentation", size=16, weight=ft.FontWeight.BOLD),
        ft.Text("      K-Means", size=14, weight=ft.FontWeight.BOLD),
        kmeans_slider,
    ], spacing=10)

    # Framing Section
    frame_controls = ft.Container(
        content=ft.Row([
            ft.ElevatedButton("Crop", on_click=lambda e: crop_image(e)),
            ft.ElevatedButton("Rotate", on_click=lambda e: rotate_image(e)),
            ft.ElevatedButton("Move", on_click=lambda e: move_image(e)),
        ], alignment=ft.MainAxisAlignment.CENTER),
        padding=10,
        visible=False,
    )

    # Show framing section
    def toggle_framing(e):
        frame_controls.visible = not frame_controls.visible
        page.snack_bar = ft.SnackBar(ft.Text("Drag to select the area."), open=True)
        page.update()

    # Functions for crop, rotate, and move
    def crop_image(e):
        print("Crop logic here")
        # Implement cropping logic using the selected rectangle
        pass

    def rotate_image(e):
        print("Rotate logic here")
        # Implement rotation logic for the selected rectangle
        pass

    def move_image(e):
        print("Move logic here")
        # Implement movement logic for the selected rectangle
        pass


    # Tools column with all controls
    tools_column = ft.ListView(
        [
            ft.ElevatedButton("Reset Image", on_click=reset_image),

            #Framing crop, rotate, and move is under the button itself
            ft.ElevatedButton("Framing", on_click=toggle_framing),
            frame_controls,

            basic_adjustments,
            color_adjustments,
            edge_detection,
            transform_controls,
            segmentation_controls,
        ],
        spacing=20,
        visible=False,
        width=300,
        height=page.window_height,
    )

    # Main layout
    page.add(
        ft.Column([
            ft.Text("Advanced Image Processing", size=24, weight=ft.FontWeight.BOLD),
            ft.ElevatedButton(
                "Upload Image",
                icon=ft.icons.UPLOAD_FILE,
                on_click=lambda _: file_picker.pick_files(
                    allowed_extensions=["png", "jpg", "jpeg", "bmp"]
                )
            ),
            ft.Row([
                tools_column,
                ft.Container(
                    content=ft.Column([
                        ft.Text("Preview", size=16, weight=ft.FontWeight.BOLD),
                        image_container
                    ]),
                    padding=10
                )
            ], spacing=20, vertical_alignment=ft.CrossAxisAlignment.START),
        ], spacing=20)
    )


    # Add keyboard shortcuts
    def on_keyboard(e: ft.KeyboardEvent):
        if e.key == "R" and e.ctrl:  # Ctrl+R to reset
            reset_image(None)
        elif e.key == "S" and e.ctrl:  # Ctrl+S to save
            save_image()

    page.on_keyboard_event = on_keyboard

    def save_image():
        if processor.current_image is None:
            page.show_snack_bar(ft.SnackBar(content=ft.Text("No image to save!")))
            return
        
        try:
            save_picker = ft.FilePicker(
                on_result=lambda e: save_image_result(e),
                allow_multiple=False
            )
            page.overlay.append(save_picker)
            save_picker.save_file(
                allowed_extensions=["png", "jpg", "jpeg"],
                file_name="processed_image.png"
            )
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error saving image: {str(ex)}")))

    def save_image_result(e):
        if e.path:
            try:
                cv2.imwrite(e.path, processor.current_image)
                page.show_snack_bar(ft.SnackBar(content=ft.Text("Image saved successfully!")))
            except Exception as ex:
                page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error saving image: {str(ex)}")))

if __name__ == "__main__":
    ft.app(target=main)