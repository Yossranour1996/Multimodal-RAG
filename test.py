import pytesseract
from PIL import Image

# Test Tesseract with a sample image
image = Image.new("RGB", (100, 100), color = (73, 109, 137))
try:
    text = pytesseract.image_to_string(image)
    print("Tesseract is working:", text)
except pytesseract.TesseractNotFoundError as e:
    print("Tesseract Not Found:", e)
