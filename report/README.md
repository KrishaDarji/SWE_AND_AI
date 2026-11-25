# Final report (ACM Transactions format, 6–9 pages)

- `main.tex`: the report. It uses `acmart` with the `acmsmall` (Transactions) layout and has the course's five section headers in order, plus the AI-disclosure section.
- `references.bib`: the bibliography.
- `build_assets.py`: builds every figure, table and in-text number from repo data, writing them to `generated/`.

## Rebuild the numbers

From the repo root:

```bash
python3 scripts/tag_histogram.py   # new code per step, from the step tags (fetch tags first: see the main README)
python3 report/build_assets.py     # writes report/generated/*.tex and *.pdf
```

Do not type numbers into `main.tex` by hand. Use the macros in `generated/numbers.tex`, so the text always matches the data.

## Compile

Upload this `report/` folder to Overleaf (it ships `acmart`), or run `pdflatex main && bibtex main && pdflatex main && pdflatex main` locally. Check the page count: the rubric deducts marks for going over 9 pages.

## Still to do

- Add the other team members (the `TODO` near `\author`).
- State any AI assistance for Treatment 7 (the `TODO` in the AI Disclosure section).
- Make the repository public, since its URL is on page one.
