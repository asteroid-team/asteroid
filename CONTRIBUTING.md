## How to contribute

The general way to contribute to Asteroid is to fork the main
repository on GitHub:
1. Fork the [main repo][asteroid] and `git clone` it.
2. Make your changes, test them, commit them and push them to your fork.
3. You can open a pull request on GitHub when you're satisfied.

If you made changes to the source code, you'll want to try them out without
installing asteroid everytime you change something.
To do that, install asteroid in develop mode either with pip
```pip install -e .[tests]``` or with python ```python setup.py develop```.

To avoid formatting roundtrips in PRs, Asteroid relies on [`black`](https://github.com/psf/black)
and [`pre-commit-hooks`](https://github.com/pre-commit/pre-commit-hooks) to handle formatting
for us. You'll need to install `requirements.txt` and install git hooks with
`pre-commit install`.

Here is a summary:

```bash
### Install
git clone your_fork_url
cd asteroid
pip install -r requirements.txt
pip install -e .
pre-commit install  # To run black before commit

# Make your changes
# Test them locally
# Commit your changes
# Push your changes
# Open a PR !
```

### Source code contributions
__All contributions to the source code of asteroid should be documented
and unit-tested__.
See [here](./tests) to run the tests with coverage reports.
Docstrings follow the [Google format][docstrings], have a look at other
docstrings in the codebase for examples. Examples in docstrings can
be bery useful, don't hesitate to add some!


### Writing new recipes.
Most new recipes should follow the standard format that is described
[here](./egs). We are not dogmatic about it, but another organization should
be explained and motivated.
We welcome any recipe on standard or new datasets, with standard or new
architectures. You can even link a paper submission with a PR number
if you'd like!

### Improving the docs.
If you found a typo, think something could be more explicit etc...
Improving the documentation is always welcome. The instructions to install
dependencies and build the docs can be found [here](./docs).
Docstrings follow the [Google format][docstrings], have a look at other
docstrings in the codebase for examples.

### Coding style
We use [PEP8 syntax conventions][pep8].
To make your life easier, we recommend running a PEP8 linter:

- Install PEP8 packages: `pip install pep8 pytest-pep8 autopep8`
- Run a standalone PEP8 check: `py.test --pep8 -m pep8`


If you have any question, [open an issue][issue] or [join the slack][slack],
we'll be happy to help you.

[asteroid]: https://github.com/mpariente/asteroid
[issue]: https://github.com/mpariente/asteroid/issues/new
[slack]: https://join.slack.com/t/asteroid-dev/shared_invite/zt-cn9y85t3-QNHXKD1Et7qoyzu1Ji5bcA
[pep8]: https://www.python.org/dev/peps/pep-0008/
[docstrings]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
