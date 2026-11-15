"""Step 2 (guess): predict objectives for unseen rows before evaluating them."""

from src.core.guess import holdout_mae, predict_row


def run(spec, rows):
    result = holdout_mae(spec, rows)
    result["example"] = {"row": rows[0], "prediction": predict_row(spec, rows[1:], rows[0])}
    return result
