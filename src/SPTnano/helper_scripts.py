import os
import scipy.io
import pandas as pd
import numpy as np
from IPython.display import Markdown, display

def generate_file_tree(startpath):
    tree = []
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree.append(f"{indent}- {os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree.append(f"{subindent}- {f}")
    return "\n".join(tree)

def display_file_tree(startpath):
    file_tree = generate_file_tree(startpath)
    display(Markdown(f"```\n{file_tree}\n```"))

def uniqueID_microns_and_seconds(df, pixelsize_microns, time_between_frames):
    '''Adds columns to the DataFrame with positions in microns and time in seconds'''
    # unique id and file identifier
    df['file_id'] = pd.Categorical(df['filename']).codes
    df['unique_id'] = df['file_id'].astype(str) + '_' + df['particle'].astype(str)
    #space transformations
    df['x_um'] = df['x'] * pixelsize_microns
    df['y_um'] = df['y'] * pixelsize_microns
    # time transformations
    df['frame_zeroed'] = df.groupby('unique_id')['frame'].transform(lambda x: x - x.iloc[0])
    df['time_s'] = df['frame'] * time_between_frames
    df['time_s_zeroed'] = df.groupby('unique_id')['time_s'].transform(lambda x: x - x.iloc[0])
    return df


# Define a function to read .mat files and convert them to DataFrames
def read_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    tr_data = mat['data']['tr'][0, 0]
    
    # Initialize an empty list to store the combined data
    combined_data = []

    # Iterate over the cells in tr_data and concatenate them
    for cell in tr_data[0]:
        if cell.size > 0:
            combined_data.append(cell)

    # Convert the combined data into a numpy array
    combined_data = np.vstack(combined_data)

    # Create a DataFrame with the appropriate columns
    # Adjust column names based on inspection
    df = pd.DataFrame(combined_data, columns=['x', 'y', 'frame', 'particle', 'column5', 'column6', 'column7', 'column8'])
    
    return df
