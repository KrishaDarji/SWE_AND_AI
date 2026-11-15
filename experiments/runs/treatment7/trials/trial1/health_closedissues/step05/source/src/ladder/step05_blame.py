"""Step 5 (blame): which inputs separate good outcomes from poor ones?"""

from src.core.blame import blame


def run(spec, rows):
    return {"findings": blame(spec, rows, top_k=5)}
