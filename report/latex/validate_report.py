"""Validate the compiled LaTeX PDF and produce private visual QA renders."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import re
import zipfile

import pymupdf
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
pdf=HERE/'main.pdf'
doc=pymupdf.open(pdf)
evidence=json.loads((HERE/'generated/evidence.json').read_text())
texts=[p.get_text() for p in doc]
all_text='\n'.join(texts)
required=['Harissh Ragav Dhamodaran','2023BCD0059','Mugupuram Kavya Amrutha','2023BCS0221',
          'Problem Statement','Method Overview with Architecture','Dataset Details',
          'Results and Visualizations','Implementation and Dataset Links',
          'References','23.19','93.84','34.46','93.61','33.54','95.51',
          '92.87','99.37','98.16','Gentlemen','businesslike','LIME','SHAP',
          'https://github.com/GL3MON/explainable-stereotype-detection','Effect of auxiliary annotations: task A versus task C','10.36','1.67']
for item in required:
    assert item in all_text,item
assert len(evidence['runs'])==10
assert len(evidence['target_classes'])==87
assert len(evidence['explanation_examples'])==2
assert evidence['overlaps']['train_test']['distinct_sentences']==4
assert not re.search(r'\bMIG\b|\bpartition\b|\[APA Style\]',all_text,re.I)
assert not re.search(r'\bmetadata\b',all_text,re.I)
for title in ['Software organization and execution environment','Canonical artifact mapping','Reproducibility of this report',
              'Training behavior and validation selection','Saved training diagnostics','Qwen task-A SFT validation history']:
    assert title not in all_text,title
comparison=evidence['metadata_comparison']
for state,corrected,introduced in [('base',242,62),('sft',53,24)]:
    item=comparison['states'][state]
    assert (item['corrected'],item['introduced'])==(corrected,introduced)
    assert item['net_additional_correct']==corrected-introduced
    for task in ('A','C'):
        matrix=item['matrices'][task]
        assert sum(sum(row) for row in matrix)==1738
        assert sum(matrix[i][i] for i in range(5))==round(evidence['runs'][f'{task}-{state}']['report']['accuracy']*1738)
log=(HERE/'main.log').read_text(errors='replace')
fatal=[s for s in log.splitlines() if any(m in s for m in ['Overfull \\hbox','Overfull \\vbox','Undefined control sequence','LaTeX Error','Package natbib Warning: Citation'])]
assert not fatal,fatal
assert 'There were undefined references' not in log
tex_files=list(HERE.glob('*.tex'))+list((HERE/'sections').glob('*.tex'))
content='\n'.join(p.read_text() for p in tex_files)
cited=set(re.findall(r'\\cite(?:p|t|year)\{([^}]+)\}',content))
bib=set(re.findall(r'\\bibitem\[[^\]]+\]\{([^}]+)\}',(HERE/'references.tex').read_text()))
assert cited<=bib,(cited-bib)
assert not any(token in content for token in ['wandb_v1_','Write detailed content','Name (Reg No)','Right click and update'])

qa=HERE/'qa';qa.mkdir(exist_ok=True)
outside=[];sparse=[];images=[]
for number,page in enumerate(doc,1):
    assert abs(page.rect.width-612)<.5 and abs(page.rect.height-792)<.5,(number,page.rect)
    if len(page.get_text().strip())<100:sparse.append(number)
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if x0<0 or y0<0 or x1>613 or y1>793:outside.append({'page':number,'text':span['text'],'bbox':span['bbox']})
    pix=page.get_pixmap(matrix=pymupdf.Matrix(1.2,1.2),alpha=False)
    im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    im.save(qa/f'page_{number:02d}.png')
    images.append(im)
assert not outside,outside
assert not sparse,sparse
for start in range(0,len(images),9):
    sheet=Image.new('RGB',(1050,1430),'#e8ebef');draw=ImageDraw.Draw(sheet)
    for offset,img in enumerate(images[start:start+9]):
        image=img.copy();image.thumbnail((330,435));x=offset%3*350+10;y=offset//3*475+24
        sheet.paste(image,(x,y));draw.text((x,y-17),f'Page {start+offset+1}',fill='black')
    sheet.save(qa/f'contact_sheet_{start//9+1}.png')
(qa/'pdf_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{text}' for i,text in enumerate(texts)))
summary={'pages':len(doc),'page_size':'US Letter; 1.25-inch left/right and 1-inch top/bottom margins in LaTeX',
         'qwen_evaluations':6,'canonical_label_reports':10,'exact_target_classes':87,
         'required_content':'pass','metric_recomputation':'pass','citations':'pass',
         'metadata_confusion_matrices':'four verified 5x5 matrices; counts and row percentages',
         'removed_sections':'5.2–5.4; References retained','partition_mentions':'none',
         'latex_errors_or_overfull_boxes':fatal,'off_page_text':outside,'sparse_pages':sparse,
         'project_repository':'https://github.com/GL3MON/explainable-stereotype-detection',
         'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}
(HERE/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
