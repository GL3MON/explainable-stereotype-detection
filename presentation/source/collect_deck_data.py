"""Collect only source-backed presentation data; no model inference is performed."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text())


def rows(relative):
    return [json.loads(line) for line in (ROOT / relative).read_text().splitlines() if line.strip()]


def normalize(value):
    return ' '.join(value.strip().casefold().split())


base_dir = 'outputs/eval/qwen35_2b_base_nothinking_20261022_173617'
sources = {
    'base_labels': f'{base_dir}/labels/evaluation.json',
    'base_json': f'{base_dir}/json/evaluation.json',
    'ft_labels': 'outputs/eval/adapter_test/evaluation.json',
    'ft_json': 'outputs/eval/json_test/evaluation.json',
    'albert': 'outputs/albert_stereodetect/evaluation.json',
    'modernbert': 'outputs/modernbert_large_lr3e5_warmup30_wd001_ep3_seed3407/evaluation.json',
}
data = {'sources': sources, 'results': {}}
for key, source in sources.items():
    value = read(source)
    assert value.get('enable_thinking', False) is False
    result = value.get('metrics', value)
    data['results'][key] = {k: result[k] for k in ('accuracy', 'macro_f1', 'weighted_f1', 'classification_report', 'confusion_matrix')}
    assert sum(map(sum, result['confusion_matrix'])) == 1738

with (ROOT / 'data/test.csv').open(encoding='utf-8-sig', newline='') as handle:
    gold = list(csv.DictReader(handle))
data['test_sha256'] = hashlib.sha256((ROOT / 'data/test.csv').read_bytes()).hexdigest()
data['splits'] = {}
for split in ('train', 'val', 'test'):
    with (ROOT / f'data/{split}.csv').open(encoding='utf-8-sig', newline='') as handle:
        data['splits'][split] = sum(1 for _ in csv.DictReader(handle))
data['class_support'] = [sum(int(row['labels']) == label for row in gold) for label in range(5)]

data['fields'] = {}
for key, path in (('base_json', f'{base_dir}/json/completions.jsonl'), ('ft_json', 'outputs/eval/json_test/completions.jsonl')):
    records = rows(path)
    assert len(records) == len(gold) == 1738
    counts = {'category': 0, 'target': 0, 'label': 0, 'all_fields': 0, 'valid_objects': 0}
    for index, (annotation, record) in enumerate(zip(gold, records)):
        assert all(annotation[k] == record[k] for k in ('Category', 'Target', 'Sentence', 'labels')), (key, index)
        try:
            prediction = json.loads(record['completion'])
        except (json.JSONDecodeError, TypeError):
            prediction = None
        if isinstance(prediction, dict):
            counts['valid_objects'] += 1
        else:
            prediction = {}
        category = prediction.get('category')
        category_ok = isinstance(category, str) and normalize(category) == normalize(annotation['Category'])
        target_ok = prediction.get('target') == annotation['Target']
        label = prediction.get('label')
        label_ok = type(label) is int and label == int(annotation['labels'])
        counts['category'] += category_ok
        counts['target'] += target_ok
        counts['label'] += label_ok
        counts['all_fields'] += category_ok and target_ok and label_ok
    data['fields'][key] = {'counts': counts, **{k: v / len(gold) for k, v in counts.items()}, 'source': path}
    assert abs(data['fields'][key]['label'] - data['results'][key]['accuracy']) < 1e-12
    record = records[0]
    data.setdefault('demo', {})[key] = {'input': record['Sentence'], 'output': json.loads(record['completion'])}
assert data['fields']['ft_json']['counts']['all_fields'] == read('outputs/eval/json_test/field_validation.json')['all_fields']['correct']

explanation_source = 'outputs/modernbert_large_lr3e5_warmup30_wd001_ep3_seed3407/explanations/explanations.jsonl'
data['explanation'] = next(row for row in rows(explanation_source) if row['test_index'] == 1713)
data['explanation_source'] = explanation_source
data['training'] = {}
for key, path in (
    ('albert', 'outputs/albert_stereodetect/run_config.json'),
    ('modernbert', 'outputs/modernbert_large_lr3e5_warmup30_wd001_ep3_seed3407/run_config.json'),
    ('qwen_labels', 'outputs/qwen35_2b_stereodetect/run_config.json'),
    ('qwen_json', 'outputs/qwen35_2b_stereodetect_json/run_config.json'),
):
    config = read(path)
    allowed = ('model', 'epochs', 'batch_size', 'gradient_accumulation', 'learning_rate', 'max_length', 'seed', 'effective_batch_size', 'warmup_steps', 'weight_decay')
    data['training'][key] = {name: config[name] for name in allowed if name in config}

data['review_results'] = [
    {'model': 'all-MiniLM-L6-v2', 'accuracy': .7733, 'macro_f1': .7429},
    {'model': 'Qwen3-Embedding-0.6B', 'accuracy': .9108, 'macro_f1': .9111},
    {'model': 'ModernBERT-large (review run)', 'accuracy': .9298, 'macro_f1': .9305},
]
data['review_source'] = 'presentation/DL-Review-2.docx'
(OUT / 'deck_data.json').write_text(json.dumps(data, indent=2) + '\n')
print('Collected metrics from full-test evaluations:', data['splits'])
print('JSON field accuracies:', json.dumps(data['fields'], indent=2))
