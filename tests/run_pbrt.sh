# Benedick scenes
# PBRT_SCENES=../pbrt_rs/data/pbrt/
# SPP=16
# for scene_dir in `ls $PBRT_SCENES`
# do
#     cargo run --example=cli --features="pbrt embree openexr" --release -- -t -1 -n $SPP -o $scene_dir.exr ../pbrt_rs/data/pbrt/$scene_dir/scene.pbrt path
# done

# PBRT scenes
PBRT_SCENES=../pbrt_rs/data/pbrt-scenes.git/pbrt-v3-scenes/
SPP=16
for scene_dir in `ls $PBRT_SCENES`
do
    for name in `ls $PBRT_SCENES/$scene_dir/*.pbrt`
    do
        img_name=$scene_dir\_`basename $name`
        echo $img_name
        cargo run --example=cli --features="pbrt embree openexr" --release -- -t -1 -n $SPP -o $img_name.exr $name path
    done
done
