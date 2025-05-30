import os
from utils import (
    RAW2RGB,
    RAW2sRGB,
    testing_record,
    extract_metadata,
)
  
def main():     
    main_folder = r"C:\Users\marin\OneDrive - UNIVERSIDAD DE SEVILLA\Escritorio\Thesis\Our dataset\Record\recordings\one_device\05.30-11.26"
    raw_folder = os.path.join(main_folder, "raw")
    image_folder = os.path.join(main_folder, "rgb")
    srgb_folder = os.path.join(main_folder, "sRGB")
    metadata_folder = os.path.join(main_folder, "metadata")
    
    print("🔄 Images extraction started.")
    RAW2RGB(raw_folder, image_folder)
    print("")

    print("🔄 sRGB extraction started.")
    RAW2sRGB(raw_folder, srgb_folder)
    print("")

    print(" Testing recording...")
    testing_record(raw_folder, image_folder, srgb_folder)
    print("")

    print("🔄 Metadata extraction started.")
    extract_metadata(raw_folder, metadata_folder)
    print("")

    print("🏁 Dataset extraction finished.")


if __name__ == "__main__":
    main()
