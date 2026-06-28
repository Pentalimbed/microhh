# Cleanup!
rm ./dom1/*
cp microhh dom1

base_dir=$(pwd) 
nproc=8

python era5_openbc_input.py --domain=1

cd ./dom1

./microhh init era5_openbc

find . -maxdepth 1 -type f -name '*_overwrite*' | while read -r file; do
    newname="${file/_overwrite/}"
    echo "Renaming: $file -> $newname"
    mv "$file" "$newname"
done

./microhh run era5_openbc

# python cross_to_nc.py -n 12

cd $base_dir
