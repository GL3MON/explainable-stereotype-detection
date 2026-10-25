"""Render the local PDF for visual review and collect readable slide text."""
import json
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

folder = Path(__file__).resolve().parent.parent
preview = folder / 'preview'
preview.mkdir(exist_ok=True)
document = pymupdf.open(folder / 'StereoDetect_BTP_Review.pdf')
assert len(document) == 24
slide_text = []
for index, page in enumerate(document):
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
    pix.save(preview / f'slide_{index+1:02d}.png')
    slide_text.append({'slide': index+1, 'text': page.get_text()})

for batch in range(3):
    sheet = Image.new('RGB', (1680, 4*270), '#e6e9ed')
    draw = ImageDraw.Draw(sheet)
    for slot in range(8):
        number = batch*8+slot+1
        image = Image.open(preview / f'slide_{number:02d}.png').convert('RGB')
        image.thumbnail((820, 235))
        x = (slot%2)*840+10
        y = (slot//2)*270+25
        sheet.paste(image, (x, y))
        draw.text((x+5, y-18), f'Slide {number}', fill='black')
    sheet.save(preview / f'contact_sheet_{batch+1}.png')

(preview / 'slide_text.json').write_text(json.dumps(slide_text, indent=2)+'\n')
print('Rendered 24 slide images and 3 contact sheets')
print('PDF text checks:', {key: any(key in page['text'] for page in slide_text) for key in ('33.54', '95.51', '34.46', '93.61', '92.87', '2.53')})
