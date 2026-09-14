#!/usr/bin/env bash
# One-command verification for the CSFR paper.
# Usage:  bash scripts/verify.sh
# Exit 0 only when code tests pass, downstream smoke test runs without
# corrupting released data, primary tables regenerate, and the manuscript
# builds with no undefined references or citations.
set -uo pipefail
cd "$(dirname "$0")/.."

MIKTEX="/c/Users/adjei/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
PDFLATEX="$MIKTEX/pdflatex.exe"
BIBTEX="$MIKTEX/bibtex.exe"
fail=0

echo "== 1. code tests =="
python -m pytest -q || fail=1

echo "== 2. downstream smoke test (temporary directory) =="
tmpdir=$(mktemp -d)
trap "rm -rf '$tmpdir'" EXIT
python scripts/run_downstream.py --config configs/downstream.yaml \
  --smoke --output-dir "$tmpdir" || fail=1

echo "== 3. regenerate primary paper tables from PSNR/SSIM CSVs =="
python scripts/update_paper_tables.py || fail=1

echo "== 4. regenerate auto tables from result CSVs =="
python scripts/render_stats_tables.py || fail=1

echo "== 5. manuscript build =="
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

echo "== 6. supplement build =="
"$PDFLATEX" -interaction=nonstopmode paper2_supplementary >/dev/null 2>&1
"$PDFLATEX" -interaction=nonstopmode paper2_supplementary >/dev/null 2>&1
serrs=$(grep -cE "^!" paper2_supplementary.log)
spages=$(grep -oE "Output written on .* \([0-9]+ pages" paper2_supplementary.log | grep -oE "[0-9]+ pages")
echo "supplement latex errors: $serrs   $spages"
[ "$serrs" -eq 0 ] || fail=1

# the files handed to a venue carry submission names, not build names. they
# were once five weeks behind the build they were copied from, which is how a
# manuscript ships with a section the repository had already fixed. refresh
# them here so a stale copy cannot outlive a green gate.
echo "== 7. refresh submission copies =="
cp paper2_reconstruction.pdf CSFR_manuscript.pdf || fail=1
cp paper2_supplementary.pdf  CSFR_supplementary.pdf || fail=1
echo "CSFR_manuscript.pdf and CSFR_supplementary.pdf refreshed from this build"
cd ..

echo "== result =="
if [ "$fail" -eq 0 ]; then echo "PASS"; else echo "FAIL"; fi
exit $fail
