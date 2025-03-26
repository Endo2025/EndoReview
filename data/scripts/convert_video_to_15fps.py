#======================================
# Import Libraries
#======================================
import argparse
import sys
import os
import pandas as pd
import shutil
from tqdm import tqdm
import subprocess

# Define function to convert video files
def convert_video(input_path, output_path, quality=20, preset="slow", fps=15):
    """
    Convert video files using ffmpeg with NVIDIA GPU acceleration.
    
    Args:
        input_path: Path to the input video file
        output_path: Path where the converted video will be saved
        quality: Constant quality value (lower is better quality)
        preset: Encoding preset (affects encoding speed vs compression efficiency)
        fps: Target frames per second for the output video
    """
    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "h264_nvenc",  # Use GPU for h264 encoding
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-preset", preset,     # Preset for quality
        "-cq", str(quality),   # Constant quality control
        "-r", str(fps),        # Framerate adjustment
        "-report",
        output_path
    ]
    subprocess.run(command, check=True) 

#======================================
# Get and set all input parameters
#======================================
def get_args_parser():
    """
    Define and parse command line arguments.
    
    Returns:
        ArgumentParser: Parser with defined arguments
    """
    parser = argparse.ArgumentParser('Convert Videoendoscopies to 15 fps', add_help=False)
    
    # Dataset parameters   
    parser.add_argument('--path_data', type=str, default=os.path.join("..","30FPS"),
                        help='Path to source video files')      

    parser.add_argument('--output_dir', type=str, default=os.path.join("..","15FPS"),
                        help='Path where converted videos will be saved')   
    
    # Dataframe
    parser.add_argument('--dataframe', type=str, default=os.path.join("..","official_splits", "videoendoscopy-classification.json"),
                        help='Path to JSON dataframe with video metadata')    
    return parser   
    
if __name__ == '__main__':
    # Parse command line arguments
    parser = get_args_parser()
    args = parser.parse_args()
    # Validate dataset directory
    if not os.path.exists(args.path_data):
        print(f"❌ Error: The specified data path '{args.path_data}' does not exist. Please check and provide the correct path.")
        sys.exit(1)
    else:
        print(f"✅ Data path verified: '{args.path_data}'")

    # Validate the existence of the official split file
    if not os.path.exists(args.dataframe):
        print(f"❌ Error: The official split file '{args.dataframe}' is missing. Please check and provide the correct path.")
        sys.exit(1)
    else:
        print(f"✅ Official split file verified: '{args.dataframe}'")


    # Load video dataframe from public dataset
    df_vd = pd.read_json(args.dataframe)

    # Process each video in the dataframe
    for idx in tqdm(df_vd.index):
        fps = df_vd["fps"].loc[idx]
  
        video_path = os.path.join(args.path_data, str(df_vd["num patient"].loc[idx]), 
                                df_vd["filename"].loc[idx])
        # Create output directory for the current patient
        save_path = os.path.join(args.output_dir, str(df_vd["num patient"].loc[idx]))
        os.makedirs(save_path, exist_ok=True)  
        # Define the output path for the processed video
        output_path = os.path.join(save_path, df_vd["filename"].loc[idx])
        # Process based on source video framerate
        if fps == 30:
            # Convert 30fps videos to 15fps
            if not os.path.exists(output_path) and os.path.exists(video_path):
                try:
                    convert_video(video_path, output_path, quality=23, preset="slow", fps=15)
                    print(f"🎥 Converted: '{video_path}' → '{output_path}' (30fps → 15fps)")
                except Exception as e:
                    print(f"❌ Error converting '{video_path}': {e}")    
        elif fps == 15:
            # Simply copy 15fps videos without conversion
            try:
                shutil.copy(video_path, output_path)
                #print(f"📂 Copied (15fps): '{video_path}' → '{output_path}'")
            except Exception as e:
                  print(f"❌ Error copying '{video_path}': {e}")
        else:
            # Warning for unexpected framerate values
            print(f"⚠️ Warning: Unknown framerate ({fps}) for '{video_path}'. Please review.")
