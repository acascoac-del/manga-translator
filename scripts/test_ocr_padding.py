import cv2
import numpy as np
from PIL import Image

def test_ocr_padding():
    # Load the original image (the one in the user's screenshot)
    img_path = r"c:\Users\chevi\demo-mimo\manga-translator-api\manga_test.jpg"
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    # We will simulate the bounding box from the CTD output for that black box
    # If the user's screenshot has it, it's roughly near the bottom.
    # We can just run CTD to get the bounding boxes.
    import asyncio
    from app.engine.detection import ctd
    
    async def run():
        await ctd.load_model()
        textlines, mask_raw, mask = await ctd.detect(img_rgb)
        print("Detected regions:", len(textlines))
        
        for i, tl in enumerate(textlines):
            x1, y1, x2, y2 = tl["bbox"]
            crop = img_rgb[y1:y2, x1:x2]
            
            # Save original crop
            cv2.imwrite(f"crop_{i}_original.jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            
            # Apply padding and inversion
            pad = 15
            h, w = img_rgb.shape[:2]
            px1 = max(0, x1 - pad)
            py1 = max(0, y1 - pad)
            px2 = min(w, x2 + pad)
            py2 = min(h, y2 + pad)
            
            crop_padded = img_rgb[py1:py2, px1:px2]
            gray_crop = cv2.cvtColor(crop_padded, cv2.COLOR_RGB2GRAY)
            if np.mean(gray_crop) < 100:
                print(f"Region {i} is DARK, inverting!")
                crop_padded = cv2.bitwise_not(crop_padded)
                
            cv2.imwrite(f"crop_{i}_padded.jpg", cv2.cvtColor(crop_padded, cv2.COLOR_RGB2BGR))
            
    asyncio.run(run())

if __name__ == "__main__":
    test_ocr_padding()
