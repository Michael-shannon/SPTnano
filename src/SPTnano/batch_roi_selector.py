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
from concurrent.futures import ThreadPoolExecutor
import tifffile as tiff
from tifffile import TiffWriter

class ROISelector:
    # Bit of a rearrange, 9-23-24#
    # def __init__(self, input_directory, output_directory, roi_width, roi_height, split_tiff=False,
    #              dark_frame_subtraction=False, dark_frame_directory=None, save_dark_corrected=False): #added split_tiff=False
    #     self.input_directory = input_directory
    #     self.output_directory = output_directory
    #     self.roi_width = roi_width
    #     self.roi_height = roi_height
    #     self.metadata = []
    #     self.current_file = None
    #     self.current_condition = None
    #     self.files = []
    #     self.conditions = []
    #     self.file_index = 0
    #     self.condition_index = 0
    #     self.split_tiff = split_tiff  # Control splitting of the TIFF
    #     self.dark_frame_subtraction = dark_frame_subtraction
    #     self.dark_frame_directory = dark_frame_directory
    #     self.save_dark_corrected = save_dark_corrected
    #     self.median_dark_frame = None  # Cache for the median dark frame
    # Bit of a rearrange, 9-23-24#

    def __init__(self, input_directory, output_directory, roi_width, roi_height, split_tiff=False,
                 dark_frame_subtraction=False, dark_frame_directory=None, save_dark_corrected=False,
                 ROI_from_metadata=False, metadata_path=None):  # New parameters added
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.split_tiff = split_tiff
        self.dark_frame_subtraction = dark_frame_subtraction
        self.dark_frame_directory = dark_frame_directory
        self.save_dark_corrected = save_dark_corrected
        self.ROI_from_metadata = ROI_from_metadata  # Boolean to enable ROI selection from metadata
        self.metadata_path = metadata_path  # Path to metadata CSV
        self.metadata_df = None  # DataFrame to hold metadata for ROI selection
        
        # Load metadata if ROI_from_metadata is enabled
        if self.ROI_from_metadata and self.metadata_path:
            self.metadata_df = pd.read_csv(self.metadata_path)

        self.metadata = []  # Metadata for the ROISelector class itself
        self.current_file = None
        self.current_condition = None
        self.files = []
        self.conditions = []
        self.file_index = 0
        self.condition_index = 0
        self.median_dark_frame = None  # Cache for the median dark frame


    def prepare_output_folders(self):
        self.conditions = [d for d in os.listdir(self.input_directory) if os.path.isdir(os.path.join(self.input_directory, d))]
        for condition in self.conditions:
            condition_output_directory = os.path.join(self.output_directory, 'data', f'Condition_{condition}')
            os.makedirs(condition_output_directory, exist_ok=True)
        os.makedirs(os.path.join(self.output_directory, 'saved_data'), exist_ok=True)
    
    def subtract_dark_frame(self, image_stack):
        if self.median_dark_frame is None:
            self.extract_median_dark_frame()  # Extract and cache the median dark frame

        # Subtract the median dark frame from the image stack
        corrected_stack = image_stack - self.median_dark_frame
        corrected_stack[corrected_stack < 0] = 0  # Ensure no negative values
        return corrected_stack

    
    def extract_median_dark_frame(self):
        # Load the dark frame stack (ND2 or TIFF)
        if self.dark_frame_directory is None:
            raise ValueError("Dark frame directory is not provided.")
        
        dark_frame_files = [f for f in os.listdir(self.dark_frame_directory) if f.lower().endswith(('nd2', 'tif', 'tiff'))]
        if not dark_frame_files:
            raise FileNotFoundError("No ND2 or TIFF files found in the dark frame directory.")

        dark_frame_path = os.path.join(self.dark_frame_directory, dark_frame_files[0])

        # Read the dark frame stack
        if dark_frame_path.lower().endswith('nd2'):
            with ND2Reader(dark_frame_path) as nd2_dark:
                dark_frames = [np.array(frame) for frame in nd2_dark]
        elif dark_frame_path.lower().endswith(('tif', 'tiff')):
            with tiff.TiffFile(dark_frame_path) as tif:
                dark_frames = [page.asarray() for page in tif.pages]

        # Calculate the median dark frame
        median_dark_frame = np.median(np.array(dark_frames), axis=0)

        # Cache the median dark frame for later use
        self.median_dark_frame = median_dark_frame    
    

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

########### Below here is the original code ########################################

    # def process_file(self, file):
    #     self.current_file = file
    #     input_filepath = os.path.join(self.input_directory, self.current_condition, file)
    #     condition_output_directory = os.path.join(self.output_directory, 'data', f'Condition_{self.current_condition}')
    #     output_filepath = os.path.join(condition_output_directory, os.path.splitext(file)[0] + '_cropped.tif')
    #     print(f"Processing file: {input_filepath}")



    #     metadata_info = {
    #         'date': '',
    #         'channels': '',
    #         'pixel_microns': '',
    #         'num_frames': ''
    #     }

    #     # Determine file type and read the image
    #     if file.lower().endswith(('tif', 'tiff')):
    #         im = Image.open(input_filepath)
    #         frames = [frame.copy() for frame in ImageSequence.Iterator(im)]
    #         first_frame = frames[0]
    #         num_frames = len(frames)
    #         metadata_info['num_frames'] = num_frames
    #     elif file.lower().endswith('nd2'):
    #         with ND2Reader(input_filepath) as nd2_file:
    #             first_frame = np.array(nd2_file[0])
    #             first_frame = Image.fromarray(first_frame)
    #             frames = [np.array(frame) for frame in nd2_file]
    #             num_frames = nd2_file.sizes.get('t', 0)
    #             # Extract relevant metadata from ND2 file
    #             nd2_metadata = nd2_file.metadata
    #             # display(nd2_metadata)  # Display ND2 metadata
                
    #             # Format date and channels
    #             if 'date' in nd2_metadata:
    #                 formatted_date = nd2_metadata['date'].strftime('%Y-%m-%d')
    #             else:
    #                 formatted_date = 'No date available'

    #             channels_str = ', '.join(nd2_metadata.get('channels', []))

    #             metadata_info = {
    #                 'date': formatted_date,
    #                 'channels': channels_str,
    #                 'pixel_microns': nd2_metadata.get('pixel_microns', 'No pixel size available'),
    #                 'num_frames': num_frames
    #             }
    #             # display(metadata_info)  # Display extracted metadata

    #     # Convert first frame to numpy array for processing
    #     first_frame_array = np.array(first_frame)

    #     # Autoscale and invert the image
    #     p2, p98 = np.percentile(first_frame_array, (2, 98))
    #     first_frame_array = exposure.rescale_intensity(first_frame_array, in_range=(p2, p98))
    #     first_frame_array = np.invert(first_frame_array)

    #     # Convert back to PIL Image
    #     first_frame = Image.fromarray(first_frame_array)

    #     # Get image dimensions
    #     width, height = first_frame.size

    #     # Function to display the image and the ROI
    #     def display_image_with_roi(x, y):
    #         fig, ax = plt.subplots(figsize=(8, 8))
    #         ax.imshow(first_frame, cmap='gray')
    #         ax.set_title(f'Select a central point for ROI - {file}')
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Y')

    #         # Draw ROI
    #         roi = plt.Rectangle((x - self.roi_width // 2, y - self.roi_height // 2), self.roi_width, self.roi_height, edgecolor='r', facecolor='none')
    #         ax.add_patch(roi)
    #         plt.show()

    #     # Create interactive widgets for selecting the ROI center
    #     x_widget = widgets.IntSlider(value=width // 2, min=0, max=width, step=1, description='X Center:')
    #     y_widget = widgets.IntSlider(value=height // 2, min=0, max=height, step=1, description='Y Center:')
        
    #     # Use the interact function to update the plot based on the widget values
    #     interact(display_image_with_roi, x=x_widget, y=y_widget)

    #     def on_button_clicked(b):
    #         center_x = x_widget.value
    #         center_y = y_widget.value
    #         clear_output(wait=True)
    #         print(f"ROI center set to: ({center_x}, {center_y}), size: ({self.roi_width}, {self.roi_height}) for file: {file}")

    #         # Calculate the cropping box based on the selected central point and size
    #         box = (center_x - self.roi_width // 2, center_y - self.roi_height // 2, center_x + self.roi_width // 2, center_y + self.roi_height // 2)

    #         # Process and save frames individually
    #         if file.lower().endswith(('tif', 'tiff')): # original line
    #             cropped_ims = [] # original line
    #             for im_frame in tqdm(frames, desc="Processing TIFF frames"): # original line
    #                 im_frame = im_frame.crop(box) # original line
    #                 cropped_ims.append(im_frame) # original line

    #             if self.split_tiff and len(cropped_ims) > 1:
    #                 # Split the stack into two parts
    #                 mid_point = len(cropped_ims) // 2

    #                 # Create arrays for p1 and p2
    #                 p1_frames = [np.array(im_frame) for im_frame in cropped_ims[:mid_point]]
    #                 p2_frames = [np.array(im_frame) for im_frame in cropped_ims[mid_point:]]

    #                 # Define the output file paths
    #                 output_filepath_p1 = os.path.splitext(output_filepath)[0] + '_p1.tif'
    #                 output_filepath_p2 = os.path.splitext(output_filepath)[0] + '_p2.tif'

    #                 # Save each part separately with the option for OME-TIFF
    #                 self._save_large_tiff(output_filepath_p1, p1_frames)
    #                 self._save_large_tiff(output_filepath_p2, p2_frames)
    #             else:
    #                 # Save as a single TIFF if no splitting is required
    #                 tiff.imwrite(output_filepath, [np.array(im_frame) for im_frame in cropped_ims], photometric='minisblack')
                
    #             cropped_ims.clear()  # Clear the list to free memory

    #         elif file.lower().endswith('nd2'):
    #             cropped_frames = []
    #             with ThreadPoolExecutor() as executor:
    #                 cropped_frames = list(tqdm(executor.map(lambda frame: Image.fromarray(frame).crop(box), frames), desc="Processing ND2 frames", total=num_frames))
    #             tiff.imwrite(output_filepath, [np.array(frame) for frame in cropped_frames], photometric='minisblack')
    #             cropped_frames.clear()  # Clear the list to free memory
    #             print(f"Finished saving {output_filepath}.")

    #         # Append metadata
    #         self.metadata.append({
    #             'Condition': self.current_condition,
    #             'Filename': file,
    #             'ROI_Center_X': center_x,
    #             'ROI_Center_Y': center_y,
    #             'ROI_Width': self.roi_width,
    #             'ROI_Height': self.roi_height,
    #             'Date': metadata_info['date'],
    #             'Channels': metadata_info['channels'],
    #             'Pixel_Microns': metadata_info['pixel_microns'],
    #             'Num_Frames': metadata_info['num_frames']
    #         })

    #         # Save metadata after each file
    #         self.save_metadata()

    #         self.file_index += 1
    #         self.process_next_file()

    #     button = widgets.Button(description=f"Set ROI")
    #     button.on_click(on_button_clicked)

        # # Display the button
        # display(button)


############################# above here is the original code ########################################
############################# Below here is with dark correction ########################################

    # def process_file(self, file):
    #     self.current_file = file
    #     input_filepath = os.path.join(self.input_directory, self.current_condition, file)
    #     condition_output_directory = os.path.join(self.output_directory, 'data', f'Condition_{self.current_condition}')
    #     output_filepath = os.path.join(condition_output_directory, os.path.splitext(file)[0] + '_cropped.tif')
    #     print(f"Processing file: {input_filepath}")

    #     metadata_info = {
    #         'date': '',
    #         'channels': '',
    #         'pixel_microns': '',
    #         'num_frames': ''
    #     }
        
    #     # Determine file type and read the image
    #     if file.lower().endswith(('tif', 'tiff')):
    #         im = Image.open(input_filepath)
    #         frames = [frame.copy() for frame in ImageSequence.Iterator(im)]
    #         first_frame = frames[0]
    #         num_frames = len(frames)
    #         metadata_info['num_frames'] = num_frames
    #     elif file.lower().endswith('nd2'):
    #         with ND2Reader(input_filepath) as nd2_file:
    #             first_frame = np.array(nd2_file[0])
    #             first_frame = Image.fromarray(first_frame)
    #             frames = [np.array(frame) for frame in nd2_file]
    #             num_frames = nd2_file.sizes.get('t', 0)
    #             # Extract relevant metadata from ND2 file
    #             nd2_metadata = nd2_file.metadata
                    
    #             # Format date and channels
    #             if 'date' in nd2_metadata:
    #                 formatted_date = nd2_metadata['date'].strftime('%Y-%m-%d')
    #             else:
    #                 formatted_date = 'No date available'

    #             channels_str = ', '.join(nd2_metadata.get('channels', []))

    #             metadata_info = {
    #                 'date': formatted_date,
    #                 'channels': channels_str,
    #                 'pixel_microns': nd2_metadata.get('pixel_microns', 'No pixel size available'),
    #                 'num_frames': num_frames
    #             }

    #     # Apply dark frame subtraction if enabled
    #     if self.dark_frame_subtraction:
    #         frames = self.subtract_dark_frame(np.array(frames))
    #         print("Dark frame subtraction applied.")
    #         # Optionally save the dark-corrected file before cropping
    #         if self.save_dark_corrected:
    #             dark_corrected_output = os.path.splitext(input_filepath)[0] + '_dark_corrected.tif'
    #             tiff.imwrite(dark_corrected_output, frames, photometric='minisblack')
    #             print(f"Dark-corrected file saved to: {dark_corrected_output}")

    #     # Convert first frame to numpy array for processing
    #     first_frame_array = np.array(first_frame)

    #     # Autoscale and invert the image
    #     p2, p98 = np.percentile(first_frame_array, (2, 98))
    #     first_frame_array = exposure.rescale_intensity(first_frame_array, in_range=(p2, p98))
    #     first_frame_array = np.invert(first_frame_array)

    #     # Convert back to PIL Image
    #     first_frame = Image.fromarray(first_frame_array)

    #     # Get image dimensions
    #     width, height = first_frame.size






    #     # Function to display the image and the ROI
    #     def display_image_with_roi(x, y):
    #         fig, ax = plt.subplots(figsize=(8, 8))
    #         ax.imshow(first_frame, cmap='gray')
    #         ax.set_title(f'Select a central point for ROI - {file}')
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Y')

    #         # Draw ROI
    #         roi = plt.Rectangle((x - self.roi_width // 2, y - self.roi_height // 2), self.roi_width, self.roi_height, edgecolor='r', facecolor='none')
    #         ax.add_patch(roi)
    #         plt.show()

    #     # Create interactive widgets for selecting the ROI center
    #     x_widget = widgets.IntSlider(value=width // 2, min=0, max=width, step=1, description='X Center:')
        # y_widget = widgets.IntSlider(value=height // 2, min=0, max=height, step=1, description='Y Center:')
        
    #     # Use the interact function to update the plot based on the widget values
    #     interact(display_image_with_roi, x=x_widget, y=y_widget)


############################# Above here is with dark correction ########################################

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
                    
                # Format date and channels
                if 'date' in nd2_metadata:
                    formatted_date = nd2_metadata['date'].strftime('%Y-%m-%d')
                else:
                    formatted_date = 'No date available'

                channels_str = ', '.join(nd2_metadata.get('channels', []))

                metadata_info = {
                    'date': formatted_date,
                    'channels': channels_str,
                    'pixel_microns': nd2_metadata.get('pixel_microns', 'No pixel size available'),
                    'num_frames': num_frames
                }

        # Apply dark frame subtraction if enabled
        if self.dark_frame_subtraction:
            frames = self.subtract_dark_frame(np.array(frames))
            print("Dark frame subtraction applied.")
            # Optionally save the dark-corrected file before cropping
            if self.save_dark_corrected:
                dark_corrected_output = os.path.splitext(input_filepath)[0] + '_dark_corrected.tif'
                tiff.imwrite(dark_corrected_output, frames, photometric='minisblack')
                print(f"Dark-corrected file saved to: {dark_corrected_output}")

        # If ROI selection from metadata is enabled
        if self.ROI_from_metadata and self.metadata_df is not None:
            # Find the metadata entry for the current file
            row = self.metadata_df[(self.metadata_df['Filename'] == file) & 
                                (self.metadata_df['Condition'] == self.current_condition)]
            
            if not row.empty:
                center_x = int(row['ROI_Center_X'].values[0])
                center_y = int(row['ROI_Center_Y'].values[0])
                roi_width = int(row['ROI_Width'].values[0])
                roi_height = int(row['ROI_Height'].values[0])
                print(f"Using ROI from metadata for file {file}: Center ({center_x}, {center_y}), Size ({roi_width}, {roi_height})")
                
                # Calculate cropping box
                box = (center_x - roi_width // 2, center_y - roi_height // 2, center_x + roi_width // 2, center_y + roi_height // 2)

                # Process and save frames
                if file.lower().endswith(('tif', 'tiff')):
                    cropped_ims = [Image.fromarray(frame).crop(box) for frame in frames]
                    tiff.imwrite(output_filepath, [np.array(im_frame) for im_frame in cropped_ims], photometric='minisblack')
                    cropped_ims.clear()
                elif file.lower().endswith('nd2'):
                    cropped_frames = []
                    with ThreadPoolExecutor() as executor:
                        cropped_frames = list(tqdm(executor.map(lambda frame: Image.fromarray(frame).crop(box), frames), desc="Processing ND2 frames", total=num_frames))
                    tiff.imwrite(output_filepath, [np.array(frame) for frame in cropped_frames], photometric='minisblack')
                    cropped_frames.clear()
                
                # Append to metadata list
                self.metadata.append({
                    'Condition': self.current_condition,
                    'Filename': file,
                    'ROI_Center_X': center_x,
                    'ROI_Center_Y': center_y,
                    'ROI_Width': roi_width,
                    'ROI_Height': roi_height,
                    'Date': metadata_info['date'],
                    'Channels': metadata_info['channels'],
                    'Pixel_Microns': metadata_info['pixel_microns'],
                    'Num_Frames': metadata_info['num_frames']
                })
                self.save_metadata()
                self.file_index += 1
                self.process_next_file()
            else:
                print(f"No metadata found for file: {file}")
                self.file_index += 1
                self.process_next_file()

        else:
            # Fallback to the interactive ROI selection method (if not using metadata)
            print("No metadata file found or ROI_from_metadata is False. Using interactive ROI selection.")
            
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
                    im_frame = Image.fromarray(im_frame).crop(box)
                    cropped_ims.append(im_frame)
                tiff.imwrite(output_filepath, [np.array(im_frame) for im_frame in cropped_ims], photometric='minisblack')
                cropped_ims.clear()  # Clear the list to free memory

            elif file.lower().endswith('nd2'):
                cropped_frames = []
                with ThreadPoolExecutor() as executor:
                    cropped_frames = list(tqdm(executor.map(lambda frame: Image.fromarray(frame).crop(box), frames), desc="Processing ND2 frames", total=num_frames))
                tiff.imwrite(output_filepath, [np.array(frame) for frame in cropped_frames], photometric='minisblack')
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

            self.file_index += 1
            self.process_next_file()

        if not self.ROI_from_metadata:
            button = widgets.Button(description=f"Set ROI")
            button.on_click(on_button_clicked)

            # Display the button
            display(button)

        # self.file_index += 1 # This added as well! 9-23-24

    # def _save_tiff(self, filepath, frames):
    #     if self.save_as_ome_tiff:
    #         imwrite(filepath, frames, photometric='minisblack', metadata={'axes': 'TYX'}, ome=True)
    #     else:
    #         imwrite(filepath, frames, photometric='minisblack')
    def _save_large_tiff(self, filepath, frames):
        with TiffWriter(filepath, bigtiff=True) as tif_writer:
            for frame in frames:
                tif_writer.write(frame, photometric='minisblack')

    def save_metadata(self):
        saved_data_dir = os.path.join(self.output_directory, 'saved_data')
        os.makedirs(saved_data_dir, exist_ok=True)
        metadata_df = pd.DataFrame(self.metadata)
        metadata_df.to_csv(os.path.join(saved_data_dir, 'metadata_summary.csv'), index=False)
        print("Metadata saved to CSV.")

def process_directory(input_directory, output_directory, roi_width, roi_height):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    selector = ROISelector(input_directory, output_directory, roi_width, roi_height)
    selector.prepare_output_folders()
    selector.process_conditions()


