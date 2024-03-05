Everything of interest is within the nesy_verification folder.
Dependencies are in pyproject.toml.

## Verification

### Setup

```bash
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