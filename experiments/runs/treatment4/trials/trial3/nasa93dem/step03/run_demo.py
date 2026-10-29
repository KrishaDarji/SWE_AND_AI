from source.nasa93_certify import certify_csv
import sys
result = certify_csv(sys.argv[1], float(sys.argv[2]))
for key, value in result.items(): print(f"{key}: {value}")
