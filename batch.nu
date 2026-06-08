#!/usr/bin/env nu
# paletti-batch.nu — apply every palette to every input image, in parallel.
#
# Progress `[n/total]` is counted in the single-threaded stage *after* par-each,
# so there is no shared mutable state and no race. Liveness depends on par-each
# streaming its results; if your build collects them, the lines batch at the end
# (still correct). Verify streaming with:
#   1..5 | par-each { |i| sleep ($i * 1sec); $i } | each { |x| print -e $x }
# If it batches and you want guaranteed-live output, move the print into the
# par-each closure (see commented line) and drop the enumerate/each stage.

def main [
    --in-dir: string = "input"        # source image directory
    --pal-dir: string = "palettes"    # palette image directory
    --out-dir: string = "output/add"  # output directory
    --threads: int = 16               # parallel workers
] {
    let inputs   = (glob $"($in_dir)/*.{jpeg,png,webp,jpg}")
    let palettes = (glob $"($pal_dir)/*.{jpeg,png,webp,jpg}")

    if ($inputs   | is-empty) { error make { msg: $"no input images in '($in_dir)'" } }
    if ($palettes | is-empty) { error make { msg: $"no palette images in '($pal_dir)'" } }

    mkdir $out_dir

    let pairs = ($inputs | each { |g| $palettes | each { |f| { g: $g, f: $f } } } | flatten)
    let total = ($pairs | length)
    let start = (date now)
    print -e $"processing ($total) combinations \(($inputs | length) inputs × ($palettes | length) palettes) on ($threads) threads"

    $pairs
    | par-each --threads $threads { |p|
        let base = ($p.g | path parse | get stem)
        let stem = ($p.f | path parse | get stem)
        let dst  = $"($out_dir)/($base)-($stem).png"

        #uv run paletti $p.g $dst -p $p.f --dither texture --rgb --texture stipple-unified.png --scale 0.006 --antialias 0.5 --denoise 0.03 
        # variants — swap in as needed:
        #uv run paletti $p.g $"($out_dir)/($base)-($stem)-ehb.png" -p $p.f --dither texture --texture screentonesdf.png --scale 0.4 --antialias 0.3 --denoise 0.03 --extra-half-brite
        uv run paletti $p.g $dst -p $p.f --blend

        # guaranteed-live alternative: uncomment, then remove the enumerate/each stage below.
        # print -e $"done  ($base)-($stem)"
        $"($base)-($stem)"
      }
    | enumerate
    | each { |it| print -e $"[($it.index + 1)/($total)] ($it.item)"; $it.item }
    | ignore

    print -e $"done in ((date now) - $start)"
}
