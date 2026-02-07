from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder(text, filename, color):
    img = Image.new('RGB', (200, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    # Draw circle
    d.ellipse([10, 10, 190, 190], outline=color, width=5)
    # Draw Text (simple approximation of centering)
    # Load default font is not always pretty, but works for placeholder
    d.text((50, 90), text, fill=color, align="center")
    
    # Ensure assets dir exists
    if not os.path.exists("assets"):
        os.makedirs("assets")
        
    img.save(os.path.join("assets", filename))
    print(f"Created {filename}")

create_placeholder("UCM", "ucm.png", "#1e40af")      # Blue for UCM
create_placeholder("BIO", "bio_icon.png", "#16a34a") # Green for Bio
