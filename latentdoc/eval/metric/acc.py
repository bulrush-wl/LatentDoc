from typing import List
from typing import Optional

def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05,
                        str_relaxed: bool = False) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        if str_relaxed:
            return target.lower() in prediction.lower() 
        else:
            return prediction.lower() == target.lower()

def Is_correct(targets: List[str] , prediction: str, numeric_toleration=0.5, str_relaxed=False):
    '''
    args:
        targets: 答案可能有多个，因此是一个list
        prediction: 模型预测的结果
        numeric_toleration: 考虑到可能预测小数，如果小数相差不超过toleration，则认为正确
        str_relaxed: 是否采用较为宽松的方式来评价字符串是否相等，即如果gt在预测中，则认为正确

    return:
        true for 1, false for 0
    '''

    correct_list = [ relaxed_correctness(t, prediction, max_relative_change=numeric_toleration, str_relaxed=str_relaxed) for t in targets]
    if any(correct_list):
        return 1
    else:
        return 0

if __name__ == '__main__':
    targets = ['abc', 'bac', 'ccv', '0.5']
    pred = 'abcdedf'
    print(Is_correct(targets, pred, str_relaxed=True))