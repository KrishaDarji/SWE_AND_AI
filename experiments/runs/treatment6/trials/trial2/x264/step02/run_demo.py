import sys
from source.x264_summary import evaluate, render_prediction
path = sys.argv[1]
r = evaluate(path)
config = {name: "0" for name in r["settings"]}
print(render_prediction(r, config))
