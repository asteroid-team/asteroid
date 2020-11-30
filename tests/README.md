# Asteroid tests
## Running the tests locally

```bash
git clone https://github.com/asteroid-team/asteroid
cd asteroid

# install module locally
pip install -e .

# install dev deps
pip install -r requirements.txt

# run tests
py.test -v
```

## Running coverage
From `asteroid` parent directory
```bash
# Install coverage
pip install coverage
# generate coverage
coverage run --source asteroid -m py.test tests -v --doctest-modules
# print coverage stats
coverage report -m
```
