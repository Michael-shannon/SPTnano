
# from PIL import Image, ImageSequence
# import os
# import matplotlib.pyplot as plt
# import ipywidgets as widgets
# from ipywidgets import interact
# from IPython.display import display, clear_output
# from nd2reader import ND2Reader
# import numpy as np
# from skimage import exposure
# import pandas as pd

# class ROISelector:
#     def __init__(self, input_directory, output_directory, roi_width, roi_height):
#         self.input_directory = input_directory
#         self.output_directory = output_directory
#         self.roi_width = roi_width
#         self.roi_height = roi_height
#         self.metadata = []
#         self.current_file = None
#         self.current_condition = None
#         self.files = []
#         self.conditions = []
#         self.file_index = 0
#         self.condition_index = 0

#     def prepare_output_folders(self):
#         self.conditions = [d for d in os.listdir(self.input_directory) if os.path.isdir(os.path.join(self.input_directory, d))]
#         for condition in self.conditions:
#             condition_output_directory = os.path.join(self.output_directory, 'data', f'Condition_{condition}')
#             os.makedirs(condition_output_directory, exist_ok=True)
#         os.makedirs(os.path.join(self.output_directory, 'saved_data'), exist_ok=True)

#     def process_conditions(self):
#         if self.condition_index < len(self.conditions):
#             self.current_condition = self.conditions[self.condition_index]
#             condition_path = os.path.join(self.input_directory, self.current_condition)
#             self.files = [f for f in os.listdir(condition_path) if f.lower().endswith(('tif', 'tiff', 'nd2'))]
#             self.file_index = 0
#             self.process_next_file()
#         else:
#             self.save_metadata()
#             print("All conditions processed.")

#     def process_next_file(self):
#         if self.file_index < len(self.files):
#             self.process_file(self.files[self.file_index])
#         else:
#             self.condition_index += 1
#             self.process_conditions()

#     def process_file(self, file):
#         self.current_file = file
#         input_filepath = os.path.join(self.input_directory, self.current_condition, file)
#         condition_output_directory = os.path.join(self.output_directory, 'data', f'Condition_{self.current_condition}')
#         output_filepath = os.path.join(condition_output_directory, os.path.splitext(file)[0] + '_cropped.tif')
#         print(f"Processing file: {input_filepath}")

#         metadata_info = {
#             'date': '',
#             'channels': '',
#             'pixel_microns': '',
#             'num_frames': ''
#         }
        
#         # Determine file type and read the image
#         if file.lower().endswith(('tif', 'tiff')):
#             im = Image.open(input_filepath)
#             first_frame = next(ImageSequence.Iterator(im))
#         elif file.lower().endswith('nd2'):
#             with ND2Reader(input_filepath) as nd2_file:
#                 first_frame = np.array(nd2_file[0])
#                 first_frame = Image.fromarray(first_frame)
#                 # Extract relevant metadata from ND2 file
#                 nd2_metadata = nd2_file.metadata
#                 metadata_info = {
#                     'date': nd2_metadata.get('date', ''),
#                     'channels': nd2_metadata.get('channels', ''),
#                     'pixel_microns': nd2_metadata.get('pixel_microns', ''),
#                     'num_frames': nd2_file.sizes.get('t', '')
#                 }

#         # Convert first frame to numpy array for processing
#         first_frame_array = np.array(first_frame)

#         # Autoscale and invert the image
#         p2, p98 = np.percentile(first_frame_array, (2, 98))
#         first_frame_array = exposure.rescale_intensity(first_frame_array, in_range=(p2, p98))
#         first_frame_array = np.invert(first_frame_array)

#         # Convert back to PIL Image
#         first_frame = Image.fromarray(first_frame_array)

#         # Get image dimensions
#         width, height = first_frame.size

#         # Function to display the image and the ROI
#         def display_image_with_roi(x, y):
#             fig, ax = plt.subplots(figsize=(8, 8))
#             ax.imshow(first_frame, cmap='gray')
#             ax.set_title(f'Select a central point for ROI - {file}')
#             ax.set_xlabel('X')
#             ax.set_ylabel('Y')

#             # Draw ROI
#             roi = plt.Rectangle((x - self.roi_width // 2, y - self.roi_height // 2), self.roi_width, self.roi_height, edgecolor='r', facecolor='none')
#             ax.add_patch(roi)
#             plt.show()

#         # Create interactive widgets for selecting the ROI center
#         x_widget = widgets.IntSlider(value=width // 2, min=0, max=width, step=1, description='X Center:')
#         y_widget = widgets.IntSlider(value=height // 2, min=0, max=height, step=1, description='Y Center:')
        
#         # Use the interact function to update the plot based on the widget values
#         interact(display_image_with_roi, x=x_widget, y=y_widget)

#         def on_button_clicked(b):
#             center_x = x_widget.value
#             center_y = y_widget.value
#             clear_output(wait=True)
#             print(f"ROI center set to: ({center_x}, {center_y}), size: ({self.roi_width}, {self.roi_height}) for file: {file}")

#             # Calculate the cropping box based on the selected central point and size
#             box = (center_x - self.roi_width // 2, center_y - self.roi_height // 2, center_x + self.roi_width // 2, center_y + self.roi_height // 2)

#             # Crop all frames based on the selected box
#             if file.lower().endswith(('tif', 'tiff')):
#                 cropped_ims = [im_frame.crop(box) for im_frame in ImageSequence.Iterator(im)]
#                 cropped_ims[0].save(output_filepath, save_all=True, append_images=cropped_ims[1:])
#             elif file.lower().endswith('nd2'):
#                 with ND2Reader(input_filepath) as nd2_file:
#                     all_frames = [frame for frame in nd2_file]
#                     cropped_frames = [Image.fromarray(frame).crop(box) for frame in all_frames]
#                     cropped_frames[0].save(output_filepath, save_all=True, append_images=cropped_frames[1:], compression='tiff_deflate')

#             # print("After cropping:")
#             if file.lower().endswith(('tif', 'tiff')):
#                 im_cropped = Image.open(output_filepath)
#                 # for i, im_frame in enumerate(ImageSequence.Iterator(im_cropped)):
#                     # print(f"Frame {i}: {im_frame.size}")
#             elif file.lower().endswith('nd2'):
#                 im_cropped = Image.open(output_filepath)
#                 # for i, im_frame in enumerate(ImageSequence.Iterator(im_cropped)):
#                     # print(f"Frame {i}: {im_frame.size}")

#             # Append metadata
#             self.metadata.append({
#                 'Condition': self.current_condition,
#                 'Filename': file,
#                 'ROI_Center_X': center_x,
#                 'ROI_Center_Y': center_y,
#                 'ROI_Width': self.roi_width,
#                 'ROI_Height': self.roi_height,
#                 'Date': metadata_info['date'],
#                 'Channels': metadata_info['channels'],
#                 'Pixel_Microns': metadata_info['pixel_microns'],
#                 'Num_Frames': metadata_info['num_frames']
#             })

#             # Save metadata after each file
#             self.save_metadata()

#             self.file_index += 1
#             self.process_next_file()

#         button = widgets.Button(description=f"Set ROI for {file}")
#         button.on_click(on_button_clicked)

#         # Display the button
#         display(button)

#     def save_metadata(self):
#         saved_data_dir = os.path.join(self.output_directory, 'saved_data')
#         os.makedirs(saved_data_dir, exist_ok=True)
#         metadata_df = pd.DataFrame(self.metadata)
#         metadata_df.to_csv(os.path.join(saved_data_dir, 'metadata_summary.csv'), index=False)
#         # print("Metadata saved to CSV.")

# def process_directory(input_directory, output_directory, roi_width, roi_height):
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)

#     selector = ROISelector(input_directory, output_directory, roi_width, roi_height)
#     selector.prepare_output_folders()
#     selector.process_conditions()


from PIL import Image, ImageSequence
import os
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display, clear_output
from nd2reader import ND2Reader
import numpy as np
from skimage import exposure
import pandas as pd
from tqdm.notebook import tqdm

class ROISelector:
    def __init__(self, input_directory, output_directory, roi_width, roi_height):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.metadata = []
        self.current_file = None
        self.current_condition = None
        self.files = []
        self.conditions = []
        self.file_index = 0
        self.condition_index = 0

    def prepare_output_folders(self):
        self.conditions = [d for d in os.listdir(self.input_directory) if os.path.isdir(os.path.join(self.input_directory, d))]
        for condition in self.conditions:
            condition_output_directory = os.path.join(self.output_directory, 'data', f'Condition_{condition}')
            os.makedirs(condition_output_directory, exist_ok=True)
        os.makedirs(os.path.join(self.output_directory, 'saved_data'), exist_ok=True)

    def process_conditions(self):
        if self.condition_index < len(self.conditions):
            self.current_condition = self.conditions[self.condition_index]
            condition_path = os.path.join(self.input_directory, self.current_condition)
            self.files = [f for f in os.listdir(condition_path) if f.lower().endswith(('tif', 'tiff', 'nd2'))]
            self.file_index = 0
            self.process_next_file()
        else:
            self.save_metadata()
            print("All conditions processed.")

    def process_next_file(self):
        if self.file_index < len(self.files):
            self.process_file(self.files[self.file_index])
        else:
            self.condition_index += 1
            self.process_conditions()

    def process_file(self, file):
        self.current_file = file
        input_filepath = os.path.join(self.input_directory, self.current_condition, file)
        condition_output_directory = os.path.join(self.output_directory, 'data', f'Condition_{self.current_condition}')
        output_filepath = os.path.join(condition_output_directory, os.path.splitext(file)[0] + '_cropped.tif')
        print(f"Processing file: {input_filepath}")

        metadata_info = {
            'date': '',
            'channels': '',
            'pixel_microns': '',
            'num_frames': ''
        }
        
        # Determine file type and read the image
        if file.lower().endswith(('tif', 'tiff')):
            im = Image.open(input_filepath)
            frames = [frame.copy() for frame in ImageSequence.Iterator(im)]
            first_frame = frames[0]
            num_frames = len(frames)
            metadata_info['num_frames'] = num_frames
        elif file.lower().endswith('nd2'):
            with ND2Reader(input_filepath) as nd2_file:
                first_frame = np.array(nd2_file[0])
                first_frame = Image.fromarray(first_frame)
                frames = [np.array(frame) for frame in nd2_file]
                num_frames = nd2_file.sizes.get('t', 0)
                # Extract relevant metadata from ND2 file
                nd2_metadata = nd2_file.metadata
                metadata_info = {
                    'date': nd2_metadata.get('date', ''),
                    'channels': nd2_metadata.get('channels', ''),
                    'pixel_microns': nd2_metadata.get('pixel_microns', ''),
                    'num_frames': num_frames
                }

        # Convert first frame to numpy array for processing
        first_frame_array = np.array(first_frame)

        # Autoscale and invert the image
        p2, p98 = np.percentile(first_frame_array, (2, 98))
        first_frame_array = exposure.rescale_intensity(first_frame_array, in_range=(p2, p98))
        first_frame_array = np.invert(first_frame_array)

        # Convert back to PIL Image
        first_frame = Image.fromarray(first_frame_array)

        # Get image dimensions
        width, height = first_frame.size

        # Function to display the image and the ROI
        def display_image_with_roi(x, y):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(first_frame, cmap='gray')
            ax.set_title(f'Select a central point for ROI - {file}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            # Draw ROI
            roi = plt.Rectangle((x - self.roi_width // 2, y - self.roi_height // 2), self.roi_width, self.roi_height, edgecolor='r', facecolor='none')
            ax.add_patch(roi)
            plt.show()

        # Create interactive widgets for selecting the ROI center
        x_widget = widgets.IntSlider(value=width // 2, min=0, max=width, step=1, description='X Center:')
        y_widget = widgets.IntSlider(value=height // 2, min=0, max=height, step=1, description='Y Center:')
        
        # Use the interact function to update the plot based on the widget values
        interact(display_image_with_roi, x=x_widget, y=y_widget)

        def on_button_clicked(b):
            center_x = x_widget.value
            center_y = y_widget.value
            clear_output(wait=True)
            print(f"ROI center set to: ({center_x}, {center_y}), size: ({self.roi_width}, {self.roi_height}) for file: {file}")

            # Calculate the cropping box based on the selected central point and size
            box = (center_x - self.roi_width // 2, center_y - self.roi_height // 2, center_x + self.roi_width // 2, center_y + self.roi_height // 2)

            # Process and save frames individually
            if file.lower().endswith(('tif', 'tiff')):
                cropped_ims = []
                for im_frame in tqdm(frames, desc="Processing TIFF frames"):
                    im_frame = im_frame.crop(box)
                    cropped_ims.append(im_frame)
                cropped_ims[0].save(output_filepath, save_all=True, append_images=cropped_ims[1:])
                cropped_ims.clear()  # Clear the list to free memory
            elif file.lower().endswith('nd2'):
                cropped_frames = []
                with ND2Reader(input_filepath) as nd2_file:
                    for i, frame in enumerate(tqdm(nd2_file, desc="Processing ND2 frames", total=num_frames)):
                        frame = np.array(frame)
                        # frame = exposure.rescale_intensity(frame, in_range=(p2, p98)) #removed this because you definitely don't need it mate
                        # frame = np.invert(frame) # Also, this is not necessary for saving
                        frame = Image.fromarray(frame).crop(box)
                        cropped_frames.append(frame)
                    cropped_frames[0].save(output_filepath, save_all=True, append_images=cropped_frames[1:])
                    cropped_frames.clear()  # Clear the list to free memory
                    print(f"Finished saving {output_filepath}.")

            # Append metadata
            self.metadata.append({
                'Condition': self.current_condition,
                'Filename': file,
                'ROI_Center_X': center_x,
                'ROI_Center_Y': center_y,
                'ROI_Width': self.roi_width,
                'ROI_Height': self.roi_height,
                'Date': metadata_info['date'],
                'Channels': metadata_info['channels'],
                'Pixel_Microns': metadata_info['pixel_microns'],
                'Num_Frames': metadata_info['num_frames']
            })

            # Save metadata after each file
            self.save_metadata()

            # del im # Delete the image object to free memory
            # del nd2_file # Delete the ND2 file object to free memory

            self.file_index += 1
            self.process_next_file()

        button = widgets.Button(description=f"Set ROI for {file}")
        button.on_click(on_button_clicked)

        # Display the button
        display(button)

    def save_metadata(self):
        saved_data_dir = os.path.join(self.output_directory, 'saved_data')
        os.makedirs(saved_data_dir, exist_ok=True)
        metadata_df = pd.DataFrame(self.metadata)
        metadata_df.to_csv(os.path.join(saved_data_dir, 'metadata_summary.csv'), index=False)
        print("Metadata saved to CSV.") #may want to return the saved data dir

def process_directory(input_directory, output_directory, roi_width, roi_height):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    selector = ROISelector(input_directory, output_directory, roi_width, roi_height)
    selector.prepare_output_folders()
    selector.process_conditions()

