import sys
from source.x264_summary import render, summarize
print(render(summarize(sys.argv[1])))
