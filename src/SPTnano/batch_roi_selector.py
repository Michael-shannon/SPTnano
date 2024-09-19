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
    def __init__(self, input_directory, output_directory, roi_width, roi_height, split_tiff=False): #added split_tiff=False
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
        self.split_tiff = split_tiff  # Control splitting of the TIFF


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
                # display(nd2_metadata)  # Display ND2 metadata
                
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
                # display(metadata_info)  # Display extracted metadata

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
            if file.lower().endswith(('tif', 'tiff')): # original line
                cropped_ims = [] # original line
                for im_frame in tqdm(frames, desc="Processing TIFF frames"): # original line
                    im_frame = im_frame.crop(box) # original line
                    cropped_ims.append(im_frame) # original line

                if self.split_tiff and len(cropped_ims) > 1:
                    # Split the stack into two parts
                    mid_point = len(cropped_ims) // 2

                    # Create arrays for p1 and p2
                    p1_frames = [np.array(im_frame) for im_frame in cropped_ims[:mid_point]]
                    p2_frames = [np.array(im_frame) for im_frame in cropped_ims[mid_point:]]

                    # Define the output file paths
                    output_filepath_p1 = os.path.splitext(output_filepath)[0] + '_p1.tif'
                    output_filepath_p2 = os.path.splitext(output_filepath)[0] + '_p2.tif'

                    # Save each part separately with the option for OME-TIFF
                    self._save_large_tiff(output_filepath_p1, p1_frames)
                    self._save_large_tiff(output_filepath_p2, p2_frames)
                else:
                    # Save as a single TIFF if no splitting is required
                    tiff.imwrite(output_filepath, [np.array(im_frame) for im_frame in cropped_ims], photometric='minisblack')
                
                cropped_ims.clear()  # Clear the list to free memory

            elif file.lower().endswith('nd2'):
                cropped_frames = []
                with ThreadPoolExecutor() as executor:
                    cropped_frames = list(tqdm(executor.map(lambda frame: Image.fromarray(frame).crop(box), frames), desc="Processing ND2 frames", total=num_frames))
                tiff.imwrite(output_filepath, [np.array(frame) for frame in cropped_frames], photometric='minisblack')
                cropped_frames.clear()  # Clear the list to free memory
                print(f"Finished saving {output_filepath}.")





            # if file.lower().endswith(('tif', 'tiff')): # original line
            #     total_frames = len(frames)
            #     frames_per_part = total_frames // self.split_factor
            #     remaining_frames = total_frames % self.split_factor

            #     for i in range(self.split_factor):
            #         start_index = i * frames_per_part
            #         end_index = start_index + frames_per_part
            #         if i == self.split_factor - 1:  # Add remaining frames to the last part
            #             end_index += remaining_frames

            #         output_filepath_part = os.path.splitext(output_filepath)[0] + f'_p{i + 1}.tif'
                    
            #         with tiff.TiffWriter(output_filepath_part) as tif_writer:
            #             for j in range(start_index, end_index):
            #                 im_frame = frames[j].crop(box)
            #                 tif_writer.save(np.array(im_frame), photometric='minisblack')

            #     cropped_ims.clear()  # Clear the list to free memory







                ############# below is a new bit again #########
                # if self.split_factor > 1 and len(cropped_ims) > 1:
                #     # Split the stack into specified number of parts
                #     num_frames_per_part = len(cropped_ims) // self.split_factor

                #     for i in range(self.split_factor):
                #         start_frame = i * num_frames_per_part
                #         end_frame = (i + 1) * num_frames_per_part if i < self.split_factor - 1 else len(cropped_ims)
                #         part_frames = [np.array(im_frame) for im_frame in cropped_ims[start_frame:end_frame]]
                        
                #         # Define the output file path with part suffix
                #         output_filepath_part = os.path.splitext(output_filepath)[0] + f'_p{i + 1}.tif'
                        
                #         # Save the part
                #         tiff.imwrite(output_filepath_part, part_frames, photometric='minisblack')
                # else:
                #     # Save as a single TIFF if no splitting is required
                #     tiff.imwrite(output_filepath, [np.array(im_frame) for im_frame in cropped_ims], photometric='minisblack')
                
                # cropped_ims.clear()  # Clear the list to free memory           


                ###### above is a new bit again #########    

                #################### new part below ################################
                # if self.split_tiff and len(cropped_ims) > 1:
                #     # Split the stack into two parts
                #     mid_point = len(cropped_ims) // 2

                #     # Create arrays for p1 and p2
                #     p1_frames = [np.array(im_frame) for im_frame in cropped_ims[:mid_point]]
                #     p2_frames = [np.array(im_frame) for im_frame in cropped_ims[mid_point:]]

                #     # Define the output file paths
                #     output_filepath_p1 = os.path.splitext(output_filepath)[0] + '_p1.tif'
                #     output_filepath_p2 = os.path.splitext(output_filepath)[0] + '_p2.tif'

                #     # Save each part separately
                #     tiff.imwrite(output_filepath_p1, p1_frames, photometric='minisblack')
                #     tiff.imwrite(output_filepath_p2, p2_frames, photometric='minisblack')
                # else:
                #     # Save as a single TIFF if no splitting is required
                #     tiff.imwrite(output_filepath, [np.array(im_frame) for im_frame in cropped_ims], photometric='minisblack')
                
                # cropped_ims.clear()  # Clear the list to free memory




                # if self.split_tiff and len(cropped_ims) > 1:
                #     # Split the stack into two parts
                #     mid_point = len(cropped_ims) // 2
                #     output_filepath_p1 = os.path.splitext(output_filepath)[0] + '_p1.tif'
                #     output_filepath_p2 = os.path.splitext(output_filepath)[0] + '_p2.tif'
                #     tiff.imwrite(output_filepath_p1, [np.array(im_frame) for im_frame in cropped_ims[:mid_point]], photometric='minisblack')
                #     tiff.imwrite(output_filepath_p2, [np.array(im_frame) for im_frame in cropped_ims[mid_point:]], photometric='minisblack')
                # else:
                #     tiff.imwrite(output_filepath, [np.array(im_frame) for im_frame in cropped_ims], photometric='minisblack')

            #################### new part above ################################

                # tiff.imwrite(output_filepath, [np.array(im_frame) for im_frame in cropped_ims], photometric='minisblack') #this removed
                # cropped_ims.clear()  # Clear the list to free memory

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

        button = widgets.Button(description=f"Set ROI")
        button.on_click(on_button_clicked)

        # Display the button
        display(button)

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


