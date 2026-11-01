import argparse, json
from fix import fix
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('dataset'); a=p.parse_args()
    print(json.dumps(fix(a.dataset), indent=2, sort_keys=True, allow_nan=False))
