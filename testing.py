from typing import Tuple

candidates = [(5, 8), (6, 8), (7, 8), (5, 7), (6, 7), (5, 6)]
pair = (6, 7)

def clean_candidates(pair, candidates:list[Tuple[int, int]]):
    removable = []
    for _ in candidates:
        if pair[0] <= _[0] and not pair[1] <= _[0]:
            removable.append(_)
        elif pair[1] >= _[1] and not pair[0] >= _[1]:
            removable.append(_)
        elif pair[0] >= _[0] and pair[1] <= _[1]:
            removable.append(_)
    
    for _ in set(removable):
        candidates.remove(_)

    return candidates
clean_candidates(pair, candidates)
print(candidates)        