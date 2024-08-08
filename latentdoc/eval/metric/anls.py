from typing import List


# edit distance
def levenshtein_distance(s1: str, s2: str) -> int:

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        dists = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                dists.append(distances[i1])
            else:
                dists.append(1 + min((distances[i1], distances[i1 + 1], dists[-1])))
        distances = dists

    return distances[-1]

# normalized edit distance
def normalized_levenshtein_distance(s1: str, s2: str) -> float:
    dist = levenshtein_distance(s1, s2)
    length = max(len(s1.upper()), len(s2.upper()))
    return 0.0 if length == 0 else dist / length


def similatiry_function(prediction: str, gold_label: str, threshold: float) -> float:
    nl_score = normalized_levenshtein_distance(prediction, gold_label)
    return 1 - nl_score if nl_score < threshold else 0.0


def anls_score(
    prediction: str, gold_labels: List[str], threshold: float = 0.5
) -> float:

    # not case sensitive, but space sensitive
    y_pred = " ".join(prediction.strip().lower().split())

    anls_scores: List[float] = []
    for gold_label in gold_labels:

        # not case sensitive, but space sensitive
        y_true = " ".join(gold_label.strip().lower().split())

        anls_score = similatiry_function(y_pred, y_true, threshold)
        anls_scores.append(anls_score)

    score = max(anls_scores)

    return score


if __name__ == '__main__':
    s1 = 'abcd'
    s2 = 'cbd'
    pred = 'abco'
    anls_score = anls_score(pred, [s1, s2])
    print(anls_score)