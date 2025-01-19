import flet as ft
import cv2
import os
import base64
from PIL import Image as PILImage
from io import BytesIO
from datetime import datetime

import image_processor
def main(page: ft.Page):
    page.title = "Advanced Image Processing Application"
    page.padding = 20
    page.scroll = "adaptive"
    
    processor = image_processor.ImageProcessor()
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

    ###     LEFT SECTION DETAILS (ui and logic)     ###
    # Lines below are the code for panels with its sliders for Left Section 

    # Reset Image
    def reset_image(e):
        try:
            result = processor.reset_image()
            update_image_display(result)
            reset_sliders()
            page.update()
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error resetting image: {str(ex)}")))

    # Reset Sliders
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

    # Download button (Logic and Button)
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
    download_button = ft.ElevatedButton(
        "Download Image",
        icon=ft.icons.DOWNLOAD,
        on_click=download_image,
        disabled=True
    )
    
    # Basic Panel
    basic_panel = ft.Column([
        ft.Text("Basic Effects", size=16, weight=ft.FontWeight.BOLD),
        ft.ElevatedButton("Grayscale", 
                         on_click=lambda _: update_image_display(processor.to_grayscale())),
        ft.ElevatedButton("Negative", 
                         on_click=lambda _: update_image_display(processor.negative_transform())),
    ], spacing=10)

    # Brightness and Contrast
    brightness_slider = ft.Slider(      # Brightness slider
        min=-100, max=100, value=0, label="Brightness",
        on_change=lambda e: update_image_display(
            processor.adjust_brightness_contrast(
                brightness=brightness_slider.value
            )
        )
    )
    contrast_slider = ft.Slider(        # Contrast slider
        min=0.1, max=3.0, value=1.0, label="Contrast",
        on_change=lambda e: update_image_display(
            processor.adjust_brightness_contrast(
                contrast=contrast_slider.value
            )
        )
    )
    brightness_contrast_panel = ft.Column([
        ft.Text("Brightness & Contrast", size=16, weight=ft.FontWeight.BOLD),
        brightness_slider,
        contrast_slider,
    ], spacing=10)

    # Color Adjustment
    red_slider = ft.Slider(             # red slider
        min=0, max=2.0, value=1.0, label="Red",
        on_change=lambda e: update_image_display(
            processor.adjust_color_channels(
                red=red_slider.value
            )
        )
    )
    green_slider = ft.Slider(           # green slider
        min=0, max=2.0, value=1.0, label="Green",
        on_change=lambda e: update_image_display(
            processor.adjust_color_channels(
                green=green_slider.value
            )
        )
    )
    blue_slider = ft.Slider(            # blue slider
        min=0, max=2.0, value=1.0, label="Blue",
        on_change=lambda e: update_image_display(
            processor.adjust_color_channels(
                blue=blue_slider.value
            )
        )
    )
    color_adjustment_panel = ft.Column([
        ft.Text("Color Adjustment", size=16, weight=ft.FontWeight.BOLD),
        red_slider,
        green_slider,
        blue_slider,
    ], spacing=10)

    # Transforming
    rotation_slider = ft.Slider(        # rotation silder
        min=-180, max=180, value=0, label="Rotation",
        on_change=lambda e: update_image_display(
            processor.apply_transform(
                rotation=rotation_slider.value
            )
        )
    )
    scale_slider = ft.Slider(           # scale slider
        min=0.1, max=2.0, value=1.0, label="Scale",
        on_change=lambda e: update_image_display(
            processor.apply_transform(
                scale=scale_slider.value
            )
        )
    )
    transform_panel = ft.Column([
        ft.Text("Transforms", size=16, weight=ft.FontWeight.BOLD),
        rotation_slider,
        scale_slider,
        ft.Row([
            ft.IconButton(icon=ft.icons.FLIP, 
                         on_click=lambda _: update_image_display(processor.flip('horizontal'))),
            ft.IconButton(icon=ft.icons.FLIP_CAMERA_ANDROID,
                         on_click=lambda _: update_image_display(processor.flip('vertical'))),
        ], spacing=10),
    ], spacing=10)

    # Color Effects
    color_effects_panel = ft.Column([
        ft.Text("Color Effects", size=16, weight=ft.FontWeight.BOLD),
        ft.ElevatedButton("Sepia", 
                         on_click=lambda _: update_image_display(processor.apply_sepia())),
        ft.ElevatedButton("Cyanotype", 
                         on_click=lambda _: update_image_display(processor.apply_cyanotype())),
    ], spacing=10)
    
    # Image Overlay
    overlay_opacity_slider = ft.Slider( # overlay slider
        min=0, max=1, value=0.5, label="Overlay Opacity",
        on_change=lambda e: update_image_display(
            processor.overlay_image(
                float(overlay_opacity_slider.value)
            )
        )
    )
    overlay_panel = ft.Column([
        ft.Text("Image Overlay", size=16, weight=ft.FontWeight.BOLD),
        ft.ElevatedButton("Add Overlay Image", 
                         on_click=lambda _: overlay_picker.pick_files(
                             allowed_extensions=["png", "jpg", "jpeg", "bmp"]
                         )),
        overlay_opacity_slider,
    ], spacing=10)

    # Spatial Filters (Logic and Slider)
    def handle_spatial_filter(e):
        filter_type = e.control.text.lower().split()[0]
        kernel_size = int(spatial_kernel_slider.value)
        if kernel_size % 2 == 0:
            kernel_size += 1
        result = processor.apply_spatial_filter(filter_type, kernel_size)
        update_image_display(result)
    spatial_kernel_slider = ft.Slider(  # spatial slider    # masih kosong
        min=3, max=15, value=3, label="Kernel Size"
    )
    filters_panel = ft.Column([
        ft.Text("Spatial Filters", size=16, weight=ft.FontWeight.BOLD),
        spatial_kernel_slider,
        ft.Row([
            ft.ElevatedButton("Mean Filter", on_click=handle_spatial_filter),
            ft.ElevatedButton("Gaussian Filter", on_click=handle_spatial_filter),
            ft.ElevatedButton("Median Filter", on_click=handle_spatial_filter),
        ], spacing=10, wrap=True),
    ], spacing=10)

    # Edge Detection (logic and Ui)
    def handle_edge_detection(e):
        method = e.control.text.lower()
        threshold1 = float(edge_threshold1_slider.value)
        threshold2 = float(edge_threshold2_slider.value)
        result = processor.apply_edge_detection(method, threshold1, threshold2)
        update_image_display(result)
    edge_threshold1_slider = ft.Slider( # edge threshold 1 slider       # masih kosong
        min=0, max=255, value=100, label="Threshold 1"
    )
    edge_threshold2_slider = ft.Slider( # edge threshold 2 slider       # masih kosong
        min=0, max=255, value=200, label="Threshold 2"
    )
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

    # Enhancement
    gamma_slider = ft.Slider(           # gamma slider
        min=0.1, max=3.0, value=1.0, label="Gamma",
        on_change=lambda e: update_image_display(
            processor.apply_gamma_correction(
                float(gamma_slider.value)
            )
        )
    )
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

    # Segmentation
    segments_slider = ft.Slider(        # segments slider               # masih kosong
        min=2, max=8, value=3, label="Segments"
    )
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

    # Binary Operations (Logic and Sliders)
    def handle_morphological_operation(e):
        operation = e.control.text.lower()
        kernel_size = int(morph_kernel_slider.value)
        result = processor.apply_morphological_operation(operation, kernel_size)
        update_image_display(result)
    morph_kernel_slider = ft.Slider(    # morph slider                  # masih kosong
        min=3, max=15, value=3, label="Kernel Size"
    )
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

    # Border and Pading (Logics, TextFields and Sliders)
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
    border_slider = ft.Slider(
        min=0, max=50, value=1, label="Border Thickness",
        on_change=lambda e: apply_border_padding()
    )
    border_color_field = ft.TextField(
        label="Border Color (Hex)", 
        hint_text="e.g., #FF5733",
        on_change=lambda e: apply_border_padding()
    )
    padding_slider = ft.Slider(
        min=0, max=50, value=1, label="Padding",
        on_change=lambda e: apply_border_padding()
    )
    padding_color_field = ft.TextField(
        label="Padding Color (Hex)", 
        hint_text="e.g., #FFF",
        on_change=lambda e: apply_border_padding()
    )
    border_padding_panel = ft.Column([
        ft.Text("Border and Padding", size=16, weight=ft.FontWeight.BOLD),
        border_slider,
        border_color_field,
        padding_slider,
        padding_color_field,
    ], spacing=10)

    # LEFT SECTION #
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
            border_padding_panel
        ],
        spacing=20,
        visible=False,
        scroll=ft.ScrollMode.AUTO,
        height=page.window_height,
    )



    ###     MAIN LAYOUT (ui and layout)    ###
    # Lines below are the code for Main Frame 

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
    
    # Create file pickers (Logic and UI)
    def update_image_display(cv_image):
        if cv_image is None:
            return
        
        # Convert to RGB if needed
        if len(cv_image.shape) == 2:  # Grayscale
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        else:  # BGR to RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL and resize to fit within 800x600
        pil_image = PILImage.fromarray(cv_image)
        pil_image.thumbnail((800, 600))
        
        # Encode the image to Base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Update the GUI with the image
        image_container.content = ft.Image(
            src_base64=img_base64,
            fit=ft.ImageFit.CONTAIN
        )
        page.update()


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
    file_picker = ft.FilePicker(on_result=handle_file_picker_result)
    overlay_picker = ft.FilePicker(on_result=handle_overlay_picker_result)
    page.overlay.extend([file_picker, overlay_picker])

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