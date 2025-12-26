from PIL import Image
import os

tracks_dir = r"c:\Users\LOQ\Documents\demos\self_driving_car\tracks"

def fix_png(file_path):
    try:
        img = Image.open(file_path)
        # Re-saving the image without the color profile
        # Simply converting and saving often strips the offending iCCP chunk
        data = list(img.getdata())
        image_without_icc = Image.new(img.mode, img.size)
        image_without_icc.putdata(data)
        image_without_icc.save(file_path)
        print(f"✓ Fixed: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"❌ Error fixing {os.path.basename(file_path)}: {e}")

if __name__ == "__main__":
    for filename in os.listdir(tracks_dir):
        if filename.endswith(".png"):
            full_path = os.path.join(tracks_dir, filename)
            fix_png(full_path)
