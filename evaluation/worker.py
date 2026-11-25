"""Run one treatment's saved code on the shared split, in isolation, and print JSON predictions.

usage: python3 evaluation/worker.py guess|race <treatment> <trial> <dataset> [probe]
The generated code is imported read-only from experiments/runs/...; nothing in it is modified.
"""
import csv, importlib.util, io, json, math, os, re, sys, tempfile, contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = {"x264": "Performance-", "nasa93dem": "Effort-", "health_closedissues": "MRE-"}
OBJECTIVES = {"x264": ["Performance-"], "nasa93dem": ["Effort-", "Defects-", "Kloc+"],
              "health_closedissues": ["MRE-", "ACC+", "PRED40+"]}


HINTS = {"x264": ("performance", "predict", "second", "time"), "nasa93dem": ("effort", "estimat", "predict"),
         "health_closedissues": ("mre", "error", "predict")}
BAD = ("std", "uncert", "neighbor", "distance", "baseline", "spread", "count", "lower", "upper", "interval", "acc", "pred40")


class Ctx:
    def __init__(self, skill, t, tr, d):
        self.t, self.tr, self.d = t, tr, d
        step = "step02" if skill == "guess" else "step10"
        base = ROOT / "experiments/runs" / f"treatment{t}" / "trials"
        self.src = (base / f"trial{tr}" / d / step / "source")
        if t == 7:
            self.src = ROOT
        self.split = ROOT / "evaluation/splits" / d
        self.train_path = str(self.split / "train.csv")
        self.test_path = str(self.split / "test.csv")
        self.target = TARGET[d]
        self.train_rows = self._read(self.train_path)
        self.test_rows = self._read(self.test_path)
        sys.path.insert(0, str(self.src))
        self._mods = {}

    @staticmethod
    def _read(p):
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def mod(self, name):
        if name not in self._mods:
            path = self.src / (name.replace(".", "/") + ".py")
            if not path.exists():
                path = self.src / name.replace(".", "/") / "__init__.py"
            spec = importlib.util.spec_from_file_location(name, path)
            m = importlib.util.module_from_spec(spec)
            sys.modules[name] = m
            spec.loader.exec_module(m)
            self._mods[name] = m
        return self._mods[name]

    def imp(self, name):
        import importlib
        return importlib.import_module(name)

    def config(self, row, numeric=False, drop_x=True):
        out = {}
        for k, v in row.items():
            if k in OBJECTIVES[self.d] and not (self.d == "nasa93dem" and k == "Kloc+"):
                continue
            if drop_x and k.endswith("X"):
                continue
            out[k] = num(v) if numeric else v
        return out

    def pick(self, res, *keys):
        """Pull the target prediction out of whatever the program returned."""
        keys = keys or (self.target,)
        import dataclasses
        for conv in ("to_dict", "as_dict", "_asdict"):
            if hasattr(res, conv) and not isinstance(res, dict):
                res = getattr(res, conv)(); break
        else:
            if dataclasses.is_dataclass(res) and not isinstance(res, type):
                res = dataclasses.asdict(res)
            elif hasattr(res, "__dict__") and not isinstance(res, (dict, list, tuple, int, float, str)):
                res = dict(vars(res))
        if isinstance(res, (int, float)):
            return float(res)
        if isinstance(res, (list, tuple)) and res and isinstance(res[0], (int, float)):
            return float(res[0])
        if isinstance(res, dict):
            for k in keys:
                if k in res and isinstance(res[k], (int, float)):
                    return float(res[k])
            for v in res.values():
                if isinstance(v, dict):
                    try:
                        return self.pick(v, *keys)
                    except ValueError:
                        pass
        for k in keys:
            if hasattr(res, k):
                return float(getattr(res, k))
        if isinstance(res, dict):
            hints = HINTS[self.d]
            for k, v in res.items():
                kl = str(k).lower()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and any(h in kl for h in hints) \
                        and not any(b in kl for b in BAD):
                    return float(v)
        raise ValueError(f"cannot find {keys} in {res!r:.300}")


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def main():
    skill, t, tr, d = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    probe = len(sys.argv) > 5
    sys.path.insert(0, str(ROOT / "evaluation"))
    import adapters
    fn = getattr(adapters, skill).get((t, tr, d))
    if fn is None:
        print(json.dumps({"error": "no adapter"})); return
    ctx = Ctx(skill, t, tr, d)
    ctx.probe = probe
    os.chdir(tempfile.mkdtemp(prefix="eval_"))
    noise = io.StringIO()
    with contextlib.redirect_stdout(noise):
        out = fn(ctx)
    print(json.dumps(out if not probe else {"probe": repr(out)[:1500]}))


if __name__ == "__main__":
    main()
