DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "Please download USPS dataset from website and decompress it to './third_party/data-local/workdir/usps' before run this script."
echo ""

echo "unpacking usps"
python $DIR/unpack_usps.py $DIR/../workdir/usps $DIR/../images/usps/

echo "Done"