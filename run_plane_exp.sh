TARGET_TIME="1.0"
OUT=results-new/plane_exp


run() {
    cargo run --release --example=cli --features="embree openexr" -- -r independent:0 -s 0.5 -n 64 -m 0.5:0.2 -t 1 -x "texture_lights" -l $OUT/log.out -o $OUT/$2_$1.exr ../scene/planes/plane_$1.xml path_kulla -s $2
}

rm -fr $OUT
mkdir -p $OUT

run "1" "pn_best_ex"
# run "1" "pn_ex"
# run "1" "kulla_clamped_ex"

run "0_5" "pn_best_ex"
# run "0_5" "pn_ex"
# run "0_5" "kulla_clamped_ex"

run "0_25" "pn_best_ex"
# run "0_25" "pn_ex"
# run "0_25" "kulla_clamped_ex"

run "0_125" "pn_best_ex"
# run "0_125" "pn_ex"
# run "0_125" "kulla_clamped_ex"

run "0_0625" "pn_best_ex"
# run "0_0625" "pn_ex"
# run "0_0625" "kulla_clamped_ex"

run "0_03125" "pn_best_ex"
# run "0_03125" "pn_ex"
# run "0_03125" "kulla_clamped_ex"