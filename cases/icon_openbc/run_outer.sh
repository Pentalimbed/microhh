set -euo pipefail

# Cleanup!
mkdir -p test/dom0
rm -f test/dom0/*00*
rm -f test/dom0/*.nc
ln -sf ../../../../build/microhh test/dom0/microhh

base_dir=$(pwd) 
nproc=4

uv run python icon_openbc_input.py --domain=0

cd test/dom0

mpiexec -n $nproc ./microhh init icon_openbc

find . -maxdepth 1 -type f -name '*_overwrite*' | while read -r file; do
    newname="${file/_overwrite/}"
    echo "Renaming: $file -> $newname"
    mv "$file" "$newname"
done

mpiexec -n $nproc ./microhh run icon_openbc

# python cross_to_nc.py -n 12

cd $base_dir
