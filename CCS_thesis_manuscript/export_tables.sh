#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$ROOT/table_wrapper.tex"
OUTDIR="$ROOT/table_pngs"

mkdir -p "$OUTDIR"

TABLE_FILES=(
  "tables/attack_examples.tex"
  "tables/defense_effectiveness_test_100.tex"
  "tables/utility_by_backend_defense.tex"
  "tables/deployment_profile_weights.tex"
  "tables/utility_am_hm_by_defense_backend.tex"
  "tables/utility_am_hm_asr_by_defense_backend.tex"
  "tables/utility_by_suite_gemini-3-1-pro-preview.tex"
  "tables/utility_by_suite_gpt-5-mini.tex"
)

echo "Exporting ${#TABLE_FILES[@]} tables to PNGs in $OUTDIR"

for tf in "${TABLE_FILES[@]}"; do
  base="$(basename "$tf" .tex)"
  tmp="_table_${base}.tex"
  pdf_tmp="$ROOT/_table_${base}.pdf"
  pdf_out="$OUTDIR/${base}.pdf"
  png_out="$OUTDIR/${base}.png"

  echo "  - $tf -> $(basename "$png_out")"

  # Create per-table wrapper
  sed "s|TABLE_FILE|$tf|" "$WRAPPER" > "$ROOT/$tmp"

  # Compile standalone table to tight PDF
  (cd "$ROOT" && pdflatex -interaction=nonstopmode -halt-on-error "$tmp" >/dev/null)

  # Move PDF into output directory (always keep it)
  mv -f "$pdf_tmp" "$pdf_out"

  # Convert to PNG (300 dpi) if tools are available
  if command -v magick >/dev/null 2>&1; then
    # ImageMagick 7+: use `magick convert`
    magick convert -density 300 "$pdf_out" -quality 100 "$png_out"
  elif command -v convert >/dev/null 2>&1; then
    # ImageMagick 6: plain `convert`
    convert -density 300 "$pdf_out" -quality 100 "$png_out"
  elif command -v pdftoppm >/dev/null 2>&1; then
    pdftoppm -png -r 300 "$pdf_out" "${png_out%.png}"
  else
    echo "    ! Neither 'convert' nor 'pdftoppm' found; leaving PDF only in $pdf_out."
  fi

  # Clean intermediates
  rm -f "$ROOT/$tmp" "${pdf_tmp%.pdf}.aux" "${pdf_tmp%.pdf}.log"
done

echo "Done."

