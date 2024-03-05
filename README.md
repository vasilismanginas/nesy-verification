Everything of interest is within the nesy_verification folder.
Dependencies are in pyproject.toml.

## Verification

### Setup

```bash
# install the package for neural verification
git clone https://github.com/Verified-Intelligence/auto_LiRPA
cd auto_LiRPA
python setup.py install
cd ..
# go to the most up-to-date branch
git checkout restructured3
# install all dependencies
poetry env use python3.11
poetry install
poetry shell
# get verification results for the MNIST task
python nesy_verification/MNIST_controller.py
```