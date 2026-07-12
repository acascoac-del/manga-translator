import os
import shutil
import zipfile

def main():
    base_dir = r"c:\Users\chevi\demo-mimo\manga-translator-api"
    hf_dir = os.path.join(base_dir, "hf-space")
    
    # Create hf-space dir
    os.makedirs(hf_dir, exist_ok=True)
    os.makedirs(os.path.join(hf_dir, "app"), exist_ok=True)
    
    # Copy app/ directory
    if os.path.exists(os.path.join(hf_dir, "app", "app")):
        shutil.rmtree(os.path.join(hf_dir, "app", "app"))
    shutil.copytree(os.path.join(base_dir, "app"), os.path.join(hf_dir, "app", "app"), dirs_exist_ok=True)
    
    # Copy requirements.txt
    shutil.copy2(os.path.join(base_dir, "requirements.txt"), os.path.join(hf_dir, "requirements.txt"))
    
    # Zip extension
    ext_zip_path = os.path.join(hf_dir, "MangaTranslatorAR_Extension.zip")
    with zipfile.ZipFile(ext_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        ext_dir = os.path.join(base_dir, "chrome-extension")
        for root, dirs, files in os.walk(ext_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, ext_dir)
                zipf.write(abs_path, rel_path)
    
    print("HF Space prep done.")

if __name__ == "__main__":
    main()
