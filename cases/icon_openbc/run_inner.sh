set -euo pipefail

# Cleanup!
mkdir -p test/dom1
rm -f test/dom1/*00*
rm -f test/dom1/*.nc
ln -sf ../../../../build/microhh test/dom1/microhh

base_dir=$(pwd) 
nproc=4

uv run python icon_openbc_input.py --domain=1

cd test/dom1

mpiexec -n $nproc ./microhh init icon_openbc

find . -maxdepth 1 -type f -name '*_overwrite*' | while read -r file; do
    newname="${file/_overwrite/}"
    echo "Renaming: $file -> $newname"
    mv "$file" "$newname"
done

mpiexec -n $nproc ./microhh run icon_openbc

# python cross_to_nc.py -n 12

cd $base_dir
