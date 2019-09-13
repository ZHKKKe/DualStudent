DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Downloading and unpacking MNIST"
mkdir -p $DIR/../workdir
mkdir -p $DIR/../workdir/mnist
python $DIR/unpack_mnist.py $DIR/../workdir/mnist $DIR/../images/mnist/

echo "Done"