DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Downloading and unpacking usps"
mkdir -p $DIR/../workdir
mkdir -p $DIR/../workdir/usps
python $DIR/unpack_usps.py $DIR/../workdir/usps $DIR/../images/usps/

echo "Done"