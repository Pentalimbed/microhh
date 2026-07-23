cmake -B build -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DUSECUDA=TRUE -DUSEMPI=FALSE -DUSESP=TRUE -DBUILD_VIZ=ON
cd build && ninja -j 16