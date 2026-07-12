import sys
from PIL import Image

def resize_icon(input_path, output_dir):
    try:
        img = Image.open(input_path)
        # crop to square from center
        width, height = img.size
        new_dim = min(width, height)
        left = (width - new_dim) / 2
        top = (height - new_dim) / 2
        right = (width + new_dim) / 2
        bottom = (height + new_dim) / 2
        img = img.crop((left, top, right, bottom))
        
        # resize and save
        sizes = [16, 48, 128]
        for size in sizes:
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            resized.save(f"{output_dir}/icon{size}.png")
        print("Icons resized successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    resize_icon(sys.argv[1], sys.argv[2])
