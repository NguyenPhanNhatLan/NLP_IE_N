from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

def build_label_mapping(datasets: List[List[str]]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    label_set = set()
    for seq in datasets:
        for tag in seq:
            label_set.add(tag)

    label_list = sorted(label_set)
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label_list, label2id, id2label


@dataclass
class Entity:
    id: str
    type: str
    text: str
    start: int
    end: int
    conf: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"Entity(id='{self.id}', type='{self.type}', text='{self.text}', "
            f"start={self.start}, end={self.end}, conf={self.conf:.2f})"
        )
