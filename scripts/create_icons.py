import os
from PIL import Image, ImageDraw

icons_dir = os.path.join(os.path.dirname(__file__), "..", "chrome-extension", "icons")
os.makedirs(icons_dir, exist_ok=True)

sizes = [16, 48, 128]

for size in sizes:
    # Create an image with blue background and a white "T"
    img = Image.new("RGBA", (size, size), (24, 119, 242, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw simple manga speech bubble
    draw.ellipse([size * 0.1, size * 0.1, size * 0.9, size * 0.7], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=max(1, size // 24))
    # Draw bubble tail
    draw.polygon([
        (size * 0.4, size * 0.65),
        (size * 0.35, size * 0.9),
        (size * 0.6, size * 0.65)
    ], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
    
    # Clean up the inner outline for tail
    draw.polygon([
        (size * 0.41, size * 0.63),
        (size * 0.36, size * 0.88),
        (size * 0.59, size * 0.63)
    ], fill=(255, 255, 255, 255))
    
    # Draw 'T' character inside
    # Choose font size depending on image size
    # For simplicity, draw lines for 'T'
    w = max(1, size // 12)
    tx = size * 0.5
    ty1 = size * 0.25
    ty2 = size * 0.55
    tw = size * 0.2
    
    # Horizontal line of T
    draw.line([(tx - tw, ty1), (tx + tw, ty1)], fill=(24, 119, 242, 255), width=w)
    # Vertical line of T
    draw.line([(tx, ty1), (tx, ty2)], fill=(24, 119, 242, 255), width=w)
    
    img.save(os.path.join(icons_dir, f"icon{size}.png"))

print("Icons created successfully!")
