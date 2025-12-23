# python -m training.rules.post_rules

from typing import List

def apply_post_rules(samples: list[dict], y_pred: List[int], id2label: dict, idx_test: list) -> List[int]:
    y_pred_postrule = y_pred.copy()

    for i, pred_id in enumerate(y_pred):
        sample_idx = idx_test[i]
        sample = samples[sample_idx]
        sentence = sample.get('sentence', '')
        head_type = sample.get('head_type')
        tail_type = sample.get('tail_type')

        

        """elif head_type == 'NAME' and tail_type == 'SKIN_CONCERNS':
            y_pred_postrule[i] = [k for k, v in id2label.items() if v=='targets_skin_concerns'][0]

        elif head_type == 'NAME' and tail_type == 'ORIGIN':
            y_pred_postrule[i] = [k for k, v in id2label.items() if v=='has_origin'][0]

        elif head_type == 'NAME' and tail_type == 'BENEFITS':
            y_pred_postrule[i] = [k for k, v in id2label.items() if v=='has_benefits'][0]"""
        
        if head_type == 'NAME' and tail_type == 'INCI':
            y_pred_postrule[i] = [k for k, v in id2label.items() if v=='has_inci_name'][0]

        elif 'chiết xuất từ' in sentence and tail_type == 'ORIGIN':
            y_pred_postrule[i] = [k for k, v in id2label.items() if v=='has_origin'][0]

        elif 'nguồn gốc' in sentence and tail_type == 'ORIGIN':
            y_pred_postrule[i] = [k for k, v in id2label.items() if v=='has_origin'][0]
        
        elif 'tổng hợp' in sentence and tail_type == 'ORIGIN':
            y_pred_postrule[i] = [k for k, v in id2label.items() if v=='has_origin'][0]
        
        elif 'phù hợp' in sentence and tail_type == 'SKIN_CONCERNS':
            y_pred_postrule[i] = [k for k, v in id2label.items() if v=='targets_skin_concerns'][0]
        
        elif 'giúp' in sentence and tail_type == "BENEFITS":
            y_pred_postrule[i] = [k for k, v in id2label.items() if v=='has_benefits'][0]

        elif 'vấn đề' in sentence and tail_type == 'SKIN_CONCERNS':
            y_pred_postrule[i] = [k for k, v in id2label.items() if v=='targets_skin_concerns'][0]

        elif 'tác dụng' in sentence and tail_type == "BENEFITS":
            y_pred_postrule[i] = [k for k, v in id2label.items() if v=='has_benefits'][0]

    return y_pred_postrule
