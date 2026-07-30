#!/usr/bin/env bash
# One-command verification for the CSFR paper.
# Usage:  bash scripts/verify.sh
# Exit 0 only when code tests pass and the manuscript builds with no undefined
# references or citations.
set -uo pipefail
cd "$(dirname "$0")/.."

MIKTEX="/c/Users/adjei/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
PDFLATEX="$MIKTEX/pdflatex.exe"
BIBTEX="$MIKTEX/bibtex.exe"
fail=0

echo "== 1. code tests =="
python -m pytest -q || fail=1

echo "== 2. regenerate auto tables from result CSVs =="
python scripts/render_stats_tables.py || fail=1

echo "== 3. manuscript build =="
cd paper
"$PDFLATEX" -interaction=nonstopmode paper2_reconstruction >/dev/null 2>&1
"$BIBTEX"   paper2_reconstruction >/dev/null 2>&1
"$PDFLATEX" -interaction=nonstopmode paper2_reconstruction >/dev/null 2>&1
"$PDFLATEX" -interaction=nonstopmode paper2_reconstruction >/dev/null 2>&1

undef=$(grep -cE "Citation .* undefined|Reference .* undefined|There were undefined references" paper2_reconstruction.log)
errs=$(grep -cE "^!" paper2_reconstruction.log)
pages=$(grep -oE "Output written on .* \([0-9]+ pages" paper2_reconstruction.log | grep -oE "[0-9]+ pages")
echo "undefined refs/cites: $undef   latex errors: $errs   $pages"
[ "$undef" -eq 0 ] || fail=1
[ "$errs"  -eq 0 ] || fail=1
cd ..

echo "== result =="
if [ "$fail" -eq 0 ]; then echo "PASS"; else echo "FAIL"; fi
exit $fail
