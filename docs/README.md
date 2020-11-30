The docs built from master can be viewed [here](https://mpariente.github.io/asteroid/).
We need to work on it so, any contribution are welcome.
Our (future) template can be found
[here](https://github.com/mpariente/asteroid_sphinx_theme).

## Building the docs
To build the docs, you'll need [Sphinx](https://www.sphinx-doc.org/en/master/),
a theme and some other package
```bash
# Start by installing the required packages
cd docs/
pip install -r requirements.txt
```
Then, you can build the docs and view it
```bash
# Build the docs
make html
# View it! (Change firefox by your favorite browser)
firefox build/html/index.html
```
If you rebuild the docs, don't forget to run `make clean` before it.

You can add this to your `.bashrc`, source it and run `run_docs`
from the `docs/` folder, that'll be easier.
```bash
alias run_docs='make clean; make html; firefox build/html/index.html'
```


### Writing good docstrings

- Start with [RST and Sphinx CheatSheet](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html)
- [Cross-referencing Python objects](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects)
    - Linking to any method in the package `:func:~mymodule.myclass.myfunc`.
    - Linking to a method in the class `:meth:mymethod`.
- Writing `Returns` with multiple values
```python
def trial(pair_wise_losses):
    r"""Trial docstring

    Args:
        pair_wise_losses: there is a dot in the beginning.

    Returns
        - :class:`torch.Tensor`:
          The loss corresponding to the best permutation of size $(batch,)$. and
          if I keep typing? It works?

        - :class:`torch.Tensor`:
          Notice that those two spaces, not a tab.
    """
```
