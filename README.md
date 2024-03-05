## End-to-end verification of a temporal neurosymbolic (NeSy) system on a synthetic MNIST task.

### Installation and Setup

```bash
# clone the repo
git clone git@github.com:vasilismanginas/nesy-verification.git
cd nesy-verification
git checkout mnist
# install all dependencies
poetry env use python3.11
poetry install
# install the package for neural verification
git clone https://github.com/Verified-Intelligence/auto_LiRPA
cd auto_LiRPA
poetry run python setup.py install
cd ..
# get verification results for the MNIST task
poetry run python nesy_verification/MNIST_controller.py
```