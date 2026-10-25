"""Recompute report evidence from saved predictions; generate vector figures and TeX tables.

Run from the original StereoDetect checkout. No model inference is performed.
The generated files are included in the portable LaTeX bundle.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / 'figures'
GEN = HERE / 'generated'
FIG.mkdir(exist_ok=True)
GEN.mkdir(exist_ok=True)
LABELS = ['Anti-stereotype', 'Stereotype', 'Neutral without target term', 'Neutral with target term', 'Bias']
SHORT = ['Anti-stereotype', 'Stereotype', 'Neutral: no target', 'Neutral: target', 'Bias']
BLUE, GREEN, ORANGE = '#4472C4', '#4F8567', '#D66A1F'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.spines.top':False,
                     'axes.spines.right':False, 'pdf.fonttype':42, 'savefig.bbox':'tight'})
sources = {}


def read_json(relative):
    path = ROOT / relative
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def rows(relative):
    path = ROOT / relative
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def tex(value):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
                    '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}',
                    '^':r'\textasciicircum{}'}.get(ch,ch) for ch in str(value))


def norm(value):
    return ' '.join(value.strip().casefold().split())


def metrics(truth, predicted, classes):
    assert len(truth) == len(predicted) and len(truth)
    report = {}
    for value in classes:
        tp = sum(a == b == value for a,b in zip(truth,predicted))
        n = sum(a == value for a in truth)
        guesses = sum(b == value for b in predicted)
        p = tp / guesses if guesses else 0.
        r = tp / n if n else 0.
        report[str(value)] = {'precision':p,'recall':r,'f1-score':2*p*r/(p+r) if p+r else 0.,'support':n}
    total = len(truth)
    report['macro avg'] = {key:sum(report[str(c)][key] for c in classes)/len(classes) for key in ('precision','recall','f1-score')}
    report['weighted avg'] = {key:sum(report[str(c)][key]*report[str(c)]['support'] for c in classes)/total for key in ('precision','recall','f1-score')}
    report['accuracy'] = sum(a==b for a,b in zip(truth,predicted))/total
    for key in ('macro avg','weighted avg'):
        report[key]['support']=total
    return report


splits = {split:rows(f'data/{split}.csv') for split in ('train','val','test')}
gold = splits['test']
y = [int(row['labels']) for row in gold]
dataset = {}
sentence_sets = {}
for split, records in splits.items():
    by_sentence = defaultdict(set)
    for row in records:
        by_sentence[row['Sentence'].strip()].add(row['labels'])
    sentence_sets[split]=by_sentence
    dataset[split] = {'rows':len(records),'labels':dict(sorted(Counter(r['labels'] for r in records).items())),
                      'categories':dict(Counter(norm(r['Category']) for r in records)),
                      'targets':len(set(r['Target'] for r in records)),
                      'empty_sentences':sum(not r['Sentence'].strip() for r in records),
                      'duplicate_sentence_rows':len(records)-len(by_sentence),
                      'conflicting_sentences':sum(len(v)>1 for v in by_sentence.values())}
overlaps = {}
for a,b in [('train','val'),('train','test'),('val','test')]:
    common = set(sentence_sets[a]) & set(sentence_sets[b])
    overlaps[a+'_'+b]={'distinct_sentences':len(common),'different_label_sets':sum(sentence_sets[a][s]!=sentence_sets[b][s] for s in common)}

specs = [
 ('A-base','A: Sentence to label -- Base','outputs/eval/qwen35_2b_base_sentence_labels_nothinking_20261022_195718/predictions.csv','outputs/eval/qwen35_2b_base_sentence_labels_nothinking_20261022_195718/evaluation.json'),
 ('A-sft','A: Sentence to label -- SFT','outputs/kaggle_qwen35_sentence_lora_1022-183119/results/artifacts/test_predictions.csv','outputs/kaggle_qwen35_sentence_lora_1022-183119/results/artifacts/evaluation.json'),
 ('C-base','C: Sentence + annotations to label -- Base','outputs/eval/qwen35_2b_base_nothinking_20261022_173617/labels/predictions.csv','outputs/eval/qwen35_2b_base_nothinking_20261022_173617/labels/evaluation.json'),
 ('C-sft','C: Sentence + annotations to label -- SFT','outputs/eval/adapter_test/predictions.csv','outputs/eval/adapter_test/evaluation.json'),
 ('albert','ALBERT-base-v2','outputs/albert_stereodetect/test_predictions.csv','outputs/albert_stereodetect/evaluation.json'),
 ('modern-original','ModernBERT-base: original','outputs/modernbert_stereodetect/test_predictions.csv','outputs/modernbert_stereodetect/evaluation.json'),
 ('modern-retuned','ModernBERT-base: retuned','outputs/modernbert_stereodetect_lr5e5_cosine_ep3_seed3407/test_predictions.csv','outputs/modernbert_stereodetect_lr5e5_cosine_ep3_seed3407/evaluation.json'),
 ('modern-large','ModernBERT-large','outputs/modernbert_large_lr3e5_warmup30_wd001_ep3_seed3407/test_predictions.csv','outputs/modernbert_large_lr3e5_warmup30_wd001_ep3_seed3407/evaluation.json'),
]
runs = {}
predictions = {}
for key,title,pred_path,eval_path in specs:
    records = rows(pred_path)
    assert len(records)==len(gold)==1738
    assert all(all(row[k]==g[k] for k in g) for row,g in zip(records,gold)),key
    predicted = [int(r['prediction']) for r in records]
    report = metrics(y,predicted,list(range(5)))
    saved = read_json(eval_path)
    saved_metrics = saved.get('metrics',saved)
    assert math.isclose(report['accuracy'],saved_metrics['accuracy'],abs_tol=1e-12),key
    assert math.isclose(report['macro avg']['f1-score'],saved_metrics['macro_f1'],abs_tol=1e-12),key
    runs[key]={'title':title,'report':report,'prediction_source':pred_path,'evaluation_source':eval_path}
    predictions[key]=predicted

targets = sorted(set(r['Target'] for r in gold))
json_fields = {}
objects = {}
for state,path,eval_path in [
 ('base','outputs/eval/qwen35_2b_base_nothinking_20261022_173617/json/completions.jsonl','outputs/eval/qwen35_2b_base_nothinking_20261022_173617/json/evaluation.json'),
 ('sft','outputs/eval/json_test/completions.jsonl','outputs/eval/json_test/evaluation.json')]:
    source = ROOT / path
    sources[path]=hashlib.sha256(source.read_bytes()).hexdigest()
    records = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(records)==len(gold)
    if all('index' in row for row in records):
        records.sort(key=lambda r:r['index'])
        assert [r['index'] for r in records]==list(range(len(gold)))
    else:
        assert all('index' not in row for row in records)
    assert all(all(row[k]==g[k] for k in g) for row,g in zip(records,gold))
    decoded=[json.loads(r['completion']) for r in records]
    assert all(type(r['label']) is int and 0<=r['label']<=4 for r in decoded)
    fields={}
    for field,classes in [('category',['gender','race','profession','religion','neutral']),('target',targets),('label',list(range(5)))]:
        truth = [norm(r['Category']) for r in gold] if field=='category' else [r['Target'] for r in gold] if field=='target' else y
        pred = [norm(r[field]) if field=='category' else r[field] for r in decoded]
        fields[field]=metrics(truth,pred,classes)
    fields['all_fields_accuracy']=sum(norm(p['category'])==norm(g['Category']) and p['target']==g['Target'] and p['label']==int(g['labels']) for p,g in zip(decoded,gold))/len(gold)
    saved=read_json(eval_path);saved_metrics=saved.get('metrics',saved)
    assert math.isclose(fields['label']['accuracy'],saved_metrics['accuracy'],abs_tol=1e-12)
    assert math.isclose(fields['label']['macro avg']['f1-score'],saved_metrics['macro_f1'],abs_tol=1e-12)
    runs['B-'+state]={'title':f'B: Sentence to category + target + label -- {state.upper()}', 'report':fields['label'], 'prediction_source':path,'evaluation_source':eval_path}
    predictions['B-'+state]=[r['label'] for r in decoded]
    json_fields[state]=fields
    objects[state]=decoded

ex_path='outputs/modernbert_large_lr3e5_warmup30_wd001_ep3_seed3407/explanations/explanations.jsonl'
path=ROOT/ex_path;sources[ex_path]=hashlib.sha256(path.read_bytes()).hexdigest()
records=[json.loads(line) for line in path.read_text().splitlines()]
examples=[next(r for r in records if r['test_index']==i) for i in (1713,1205)]
for record in examples:
    assert record['sentence']==gold[record['test_index']]['Sentence']
    assert record['gold_label']==int(gold[record['test_index']]['labels'])

metadata_comparison={'num_examples':len(gold),'class_names':LABELS,'states':{}}
for state in ('base','sft'):
    matrices={}
    for task in ('A','C'):
        cm=np.zeros((5,5),dtype=int)
        for truth,pred in zip(y,predictions[f'{task}-{state}']):
            assert 0<=pred<5
            cm[truth,pred]+=1
        assert int(cm.sum())==len(gold)
        assert cm.sum(axis=1).tolist()==[y.count(i) for i in range(5)]
        assert math.isclose(float(np.trace(cm))/len(gold),runs[f'{task}-{state}']['report']['accuracy'],abs_tol=1e-12)
        matrices[task]=cm.tolist()
    a=predictions[f'A-{state}'];c=predictions[f'C-{state}']
    corrected=sum(pa!=truth and pc==truth for truth,pa,pc in zip(y,a,c))
    introduced=sum(pa==truth and pc!=truth for truth,pa,pc in zip(y,a,c))
    gain=sum(pc==truth for truth,pc in zip(y,c))-sum(pa==truth for truth,pa in zip(y,a))
    assert corrected-introduced==gain
    metadata_comparison['states'][state]={'matrices':matrices,'corrected':corrected,'introduced':introduced,'net_additional_correct':gain,
        'accuracy_gain_pp':100*gain/len(gold),'prediction_sources':{task:runs[f'{task}-{state}']['prediction_source'] for task in ('A','C')}}
(GEN/'metadata_comparison.json').write_text(json.dumps(metadata_comparison,indent=2)+'\n')
snapshot={'dataset':dataset,'overlaps':overlaps,'runs':runs,'json_fields':json_fields,'target_classes':targets,
          'explanation_examples':examples,'metadata_comparison':metadata_comparison,'source_sha256':sources}
(GEN/'evidence.json').write_text(json.dumps(snapshot,indent=2)+'\n')


def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.png',dpi=180)
    plt.close(fig)


fig,axes=plt.subplots(1,2,figsize=(6.0,2.7),layout='constrained')
x=np.arange(5);width=.24
for i,(split,color) in enumerate(zip(('train','val','test'),(BLUE,GREEN,ORANGE))):
    axes[0].bar(x+(i-1)*width,[dataset[split]['labels'][str(c)] for c in range(5)],width,label={'train':'Train','val':'Validation','test':'Test'}[split],color=color)
axes[0].set_xticks(x,[str(i) for i in range(5)]);axes[0].set_xlabel('Class ID');axes[0].set_ylabel('Number of examples');axes[0].legend(fontsize=7)
cats=['gender','race','profession','religion','neutral']
axes[1].barh(cats,[dataset['test']['categories'][c] for c in cats],color=BLUE)
for i,c in enumerate(cats):axes[1].text(dataset['test']['categories'][c]+8,i,str(dataset['test']['categories'][c]),va='center',fontsize=8)
axes[1].set_xlim(0,700);axes[1].set_xlabel('Test examples');axes[1].invert_yaxis()
save(fig,'dataset_distribution')

fig,axes=plt.subplots(1,2,figsize=(6,2.65),layout='constrained')
for ax,key,title in zip(axes,('accuracy','f1-score'),('Label accuracy (%)','Label macro F1 (%)')):
    for offset,state,color in [(-.18,'base',BLUE),(.18,'sft',GREEN)]:
        vals=[runs[f'{task}-{state}']['report']['accuracy' if key=='accuracy' else 'macro avg'] for task in 'ABC']
        vals=[v if isinstance(v,float) else v['f1-score'] for v in vals]
        bars=ax.bar(np.arange(3)+offset,np.array(vals)*100,.34,label=state.upper() if state=='sft' else 'Base',color=color)
        ax.bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
    ax.set_xticks(range(3),['A','B','C']);ax.set_ylim(0,110);ax.set_ylabel(title)
    ax.legend(fontsize=7,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
save(fig,'qwen_task_comparison')

fig,axes=plt.subplots(1,2,figsize=(6,2.8),layout='constrained')
for ax,key,title in zip(axes,('A-base','A-sft'),('Task A: Base','Task A: SFT')):
    cm=np.zeros((5,5),dtype=int)
    for truth,pred in zip(y,predictions[key]):
        if 0<=pred<5:cm[truth,pred]+=1
    ax.imshow(cm,cmap='Blues',vmin=0,vmax=400)
    for i in range(5):
        for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>210 else '#182A3E')
    ax.set_xticks(range(5));ax.set_yticks(range(5));ax.set_xlabel('Predicted class');ax.set_ylabel('Gold class');ax.set_title(title,fontsize=10)
save(fig,'task_a_confusion')

fig,axes=plt.subplots(2,2,figsize=(6.2,5.2),layout='constrained')
for row,state in enumerate(('base','sft')):
    for col,task in enumerate(('A','C')):
        ax=axes[row,col]
        cm=np.array(metadata_comparison['states'][state]['matrices'][task])
        percentages=100*cm/cm.sum(axis=1,keepdims=True)
        image=ax.imshow(percentages,cmap='Blues',vmin=0,vmax=100)
        for i in range(5):
            for j in range(5):
                color='white' if percentages[i,j]>=55 else '#182A3E'
                ax.text(j,i-.13,str(cm[i,j]),ha='center',va='center',fontsize=8,color=color)
                ax.text(j,i+.19,f'{percentages[i,j]:.1f}%',ha='center',va='center',fontsize=6.2,color=color)
        report=runs[f'{task}-{state}']['report'];label='Base' if state=='base' else 'SFT'
        inputs='Sentence only' if task=='A' else 'Gold annotations'
        ax.set_title(f'{label} · Task {task}: {inputs}\nAccuracy {report["accuracy"]*100:.2f}% | Macro F1 {report["macro avg"]["f1-score"]*100:.2f}%',fontsize=8,pad=6)
        ax.set_xticks(range(5));ax.set_yticks(range(5));ax.tick_params(labelsize=7)
        ax.set_xlabel('Predicted class',fontsize=8);ax.set_ylabel('Gold class',fontsize=8)
bar=fig.colorbar(image,ax=axes.ravel().tolist(),fraction=.03,pad=.02)
bar.set_label('Share of gold class (%)',fontsize=8);bar.ax.tick_params(labelsize=7)
save(fig,'metadata_confusion_comparison')

history=read_json('outputs/modernbert_large_lr3e5_warmup30_wd001_ep3_seed3407/trainer_state.json')['log_history']
valhist=read_json('outputs/kaggle_qwen35_sentence_lora_1022-183119/results/artifacts/validation_history.json')
fig,axes=plt.subplots(1,2,figsize=(6.0,2.5),layout='constrained')
train=[r for r in history if 'loss' in r];valid=[r for r in history if 'eval_loss' in r]
axes[0].plot([r['step'] for r in train],[r['loss'] for r in train],color=BLUE,alpha=.35,lw=.5,label='Per-step train loss')
loss=np.array([r['loss'] for r in train]);smooth=np.convolve(loss,np.ones(15)/15,mode='valid')
axes[0].plot([r['step'] for r in train][14:],smooth,color=BLUE,lw=1.3,label='15-step mean')
axes[0].plot([r['step'] for r in valid],[r['eval_loss'] for r in valid],'o-',color=ORANGE,ms=3,label='Validation loss')
axes[0].set_xlabel('Optimizer step');axes[0].set_ylabel('Cross-entropy');axes[0].set_title('ModernBERT-large',fontsize=10);axes[0].legend(fontsize=6.5)
f1s=[r['validation']['macro_f1']*100 for r in valhist]
axes[1].plot([r['epoch'] for r in valhist],f1s,'o-',color=GREEN)
for epoch,value in zip([1,2,3],f1s):axes[1].annotate(f'{value:.2f}',(epoch,value),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
axes[1].set_xticks([1,2,3]);axes[1].set_xlabel('Epoch');axes[1].set_ylabel('Validation macro F1 (%)');axes[1].set_ylim(88,99);axes[1].set_title('Qwen task A: 2×T4 SFT',fontsize=10)
save(fig,'training_history')

cmap=LinearSegmentedColormap.from_list('signed',['#2B73B9','#FAFAF8','#D66A1F'])
normalizer=Normalize(vmin=-.70,vmax=.70)
for number,record in enumerate(examples,1):
    fig=plt.figure(figsize=(6,2.4));ax=fig.add_axes([.03,.23,.94,.55]);ax.set_xlim(0,1);ax.set_ylim(0,2);ax.axis('off')
    explained=record['predicted_label'];detail=record['explanations'][str(explained)]
    words=record['words'];weights=np.array([max(4,len(w)+1) for w in words]);widths=weights/weights.sum()*.87
    for row,(method,key) in enumerate([('LIME','lime_values'),('SHAP','shap_values')]):
        left=.12;bottom=1.1-row
        ax.text(0,bottom+.35,method,va='center',fontsize=9,fontweight='bold')
        for word,width,value in zip(words,widths,detail[key]):
            ax.add_patch(Rectangle((left,bottom),width-.005,.78,facecolor=cmap(normalizer(value)),edgecolor='#CAD3DD',lw=.5))
            ax.text(left+(width-.005)/2,bottom+.51,word,ha='center',va='center',fontsize=8 if len(words)<6 else 7.2,fontweight='bold')
            ax.text(left+(width-.005)/2,bottom+.20,f'{value:+.3f}',ha='center',va='center',fontsize=7)
            left+=width
    fig.text(.03,.93,'“'+record['sentence']+'”',fontsize=10,fontweight='bold')
    fig.text(.03,.83,f"Gold {record['gold_label']} ({LABELS[record['gold_label']]})  |  Predicted {explained} ({LABELS[explained]})",fontsize=8)
    cax=fig.add_axes([.32,.14,.36,.045]);bar=fig.colorbar(plt.cm.ScalarMappable(norm=normalizer,cmap=cmap),cax=cax,orientation='horizontal',ticks=[-.70,0,.70]);bar.ax.tick_params(labelsize=7,length=2)
    fig.text(.03,.145,f'Blue: lowers class {explained} score',fontsize=7.4,color='#2B73B9')
    fig.text(.72,.145,f'Orange: raises class {explained} score',fontsize=7.4,color='#A95015')
    fig.text(.5,.025,'Near white = near zero; darker = stronger. Values are signed attributions, not word probabilities.',ha='center',fontsize=7)
    save(fig,f'explanation_{number}')


def write(name,lines):
    (GEN/name).write_text('\n'.join(lines)+'\n')


dataset_rows=[f'{i} & {tex(label)} & '+ ' & '.join(str(dataset[s]['labels'][str(i)]) for s in ('train','val','test'))+r' \\' for i,label in enumerate(LABELS)]
write('dataset_rows.tex',dataset_rows)
write('category_rows.tex',[tex(c)+' & '+' & '.join(str(dataset[s]['categories'][c]) for s in ('train','val','test'))+r' \\' for c in ('gender','race','profession','religion','neutral')])
write('qwen_rows.tex',[f"{task} & {state.title() if state=='base' else 'SFT'} & {runs[f'{task}-{state}']['report']['accuracy']*100:.2f} & {runs[f'{task}-{state}']['report']['macro avg']['f1-score']*100:.2f}"+r' \\' for task in 'ABC' for state in ('base','sft')])
write('encoder_rows.tex',[tex(runs[k]['title'])+f" & {runs[k]['report']['accuracy']*100:.2f} & {runs[k]['report']['macro avg']['f1-score']*100:.2f}"+r' \\' for k in ('albert','modern-original','modern-retuned','modern-large')])
write('json_rows.tex',[tex(field.title())+' & '+' & '.join(f'{json_fields[state][field][key]*100:.2f}' if key=='accuracy' else f"{json_fields[state][field]['macro avg']['f1-score']*100:.2f}" for state,key in [('base','accuracy'),('sft','accuracy'),('base','f1') ,('sft','f1')])+r' \\' for field in ('category','target','label')])

lines=[]
for key in ('A-base','A-sft','B-base','B-sft','C-base','C-sft','albert','modern-original','modern-retuned','modern-large'):
    r=runs[key];report=r['report']
    lines.append(r'\multicolumn{5}{l}{\textbf{'+tex(r['title'])+r'}} \\*')
    for i,label in enumerate(SHORT):
        value=report[str(i)]
        lines.append(tex(f'{i}: {label}')+' & '+' & '.join(f'{value[k]*100:.2f}' for k in ('precision','recall','f1-score'))+f" & {value['support']}"+r' \\*')
    for agg in ('macro avg','weighted avg'):
        value=report[agg]
        lines.append(tex(agg.title())+' & '+' & '.join(f'{value[k]*100:.2f}' for k in ('precision','recall','f1-score'))+f" & {value['support']}"+r' \\*')
    lines.append(r'\multicolumn{5}{l}{Accuracy: '+f"{report['accuracy']*100:.2f}"+r'\%} \\ \addlinespace[5pt]')
write('label_reports.tex',lines)
for field,classes in [('category',['gender','race','profession','religion','neutral']),('target',targets)]:
    lines=[]
    for label in classes:
        vals=[json_fields[state][field][label] for state in ('base','sft')]
        lines.append(tex(label)+' & '+' & '.join(f'{v[k]*100:.2f}' for v in vals for k in ('precision','recall','f1-score'))+f" & {vals[0]['support']}"+r' \\')
    write(f'{field}_reports.tex',lines)

write('validation_rows.tex',[f"{r['epoch']} & {r['step']} & {r['validation']['accuracy']*100:.2f} & {r['validation']['macro_f1']*100:.2f}"+r' \\' for r in valhist])
write('metadata_comparison_rows.tex',[
    ('Base' if state=='base' else 'SFT')+' & '+' & '.join([
        f"{runs[f'A-{state}']['report']['accuracy']*100:.2f}",
        f"{runs[f'C-{state}']['report']['accuracy']*100:.2f}",
        f"+{metadata_comparison['states'][state]['accuracy_gain_pp']:.2f}",
        f"{runs[f'A-{state}']['report']['macro avg']['f1-score']*100:.2f}",
        f"{runs[f'C-{state}']['report']['macro avg']['f1-score']*100:.2f}"])+r' \\' for state in ('base','sft')])
print('Verified 10 canonical label reports; 2 JSON field reports; 87 target classes; 2 saved explanation examples.')
(GEN/'evidence.json').write_text(json.dumps(snapshot,indent=2)+'\n')
print('Generated seven vector figure sets, including the four-panel annotation-conditioning comparison, plus report tables and evidence.json.')
