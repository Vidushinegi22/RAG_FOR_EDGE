import fitz  # PyMuPDF
import os
from PIL import Image
import pandas as pd
from PIL import PngImagePlugin

# Fix for multi-frame PNG handling
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024**2  # Increase max text chunk size

pdf_path = "GrandVitara 1.pdf"
doc = fitz.open(pdf_path)

output_dir = "output_dataset"
os.makedirs(output_dir, exist_ok=True)

data = []

for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()
    
    image_list = page.get_images(full=True)
    image_paths = []

    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_name = f"page{page_num+1}_img{img_index+1}.{image_ext}"
        image_path = os.path.join(output_dir, image_name)

        with open(image_path, "wb") as img_file:
            img_file.write(image_bytes)

        # Check if image is multi-frame or animated
        try:
            with Image.open(image_path) as img:
                # For animated images (like GIFs), extract the first frame
                if hasattr(img, 'is_animated') and img.is_animated:
                    img.seek(0)  # Get first frame
                    single_frame_path = os.path.join(output_dir, f"converted_{image_name}")
                    img.save(single_frame_path)
                    image_paths.append(os.path.basename(single_frame_path))
                    os.remove(image_path)  # Remove original animated image
                else:
                    # For multi-frame images (like some PNGs), extract first frame
                    if img.n_frames > 1:
                        img.seek(0)
                        single_frame_path = os.path.join(output_dir, f"converted_{image_name}")
                        img.save(single_frame_path)
                        image_paths.append(os.path.basename(single_frame_path))
                        os.remove(image_path)
                    else:
                        image_paths.append(image_name)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue  # Skip this image

    data.append({
        "page": page_num + 1,
        "text": text.strip(),
        "images": image_paths
    })

df = pd.DataFrame(data)
df.to_csv(os.path.join(output_dir, "grand_vitara_dataset.csv"), index=False)

