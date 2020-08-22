# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import glob
import shutil
import builtins
import asteroid_sphinx_theme


PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, "..", "..")
sys.path.insert(0, os.path.abspath(PATH_ROOT))

# -- Project information -----------------------------------------------------

project = "asteroid"
copyright = "2019, Oncoming"
author = "Manuel Pariente et al."
# The short X.Y version
version = "0.0.1"
# The full version, including alpha/beta/rc tags
release = "0.0.1"

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "1.4"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    # 'sphinxcontrib.mockautodoc',
    # 'sphinxcontrib.fulltoc',  # breaks pytorch-theme with unexpected
    # w argument 'titles_only'
    # We can either use viewcode, which shows source code in the doc page
    "sphinx.ext.viewcode",
    # Or linkcode to find the corresponding code in github. Start with viewcode
    # 'sphinx.ext.linkcode',
    # 'recommonmark',
    "m2r2",
    "nbsphinx",
]

# Napoleon config
napoleon_include_special_with_doc = True
napoleon_use_ivar = True
napoleon_use_rtype = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# https://berkeley-stat159-f17.github.io/stat159-f17/lectures/14-sphinx..html#conf.py-(cont.)
# https://stackoverflow.com/questions/38526888/embed-ipython-notebook-in-sphinx-document
# I execute the notebooks manually in advance. If notebooks test the code,
# they should be run at build time.
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
# source_suffix = ['.rst', '.md', '.ipynb']
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    # '.md': 'markdown',
    ".ipynb": "nbsphinx",
}

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# http://www.sphinx-doc.org/en/master/usage/theming.html#builtin-themes
# html_theme = 'bizstyle'
# https://sphinx-themes.org

html_theme = "asteroid_sphinx_theme"
html_theme_path = [asteroid_sphinx_theme.get_html_theme_path()]
#
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "pytorch_project": "docs",
    "canonical_url": "https://github.com/mpariente/asteroid",
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
}

html_logo = "_static/images/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = project + "-doc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
    # Latex figure (float) alignment
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, project + ".tex", project + " Documentation", author, "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, project, project + " Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        project,
        project + " Documentation",
        author,
        project,
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Intersphinx config
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/latest/", None),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# https://github.com/rtfd/readthedocs.org/issues/1139
# I use sphinx-apidoc to auto-generate API documentation for my project.
# Right now I have to commit these auto-generated files to my repository
# so that RTD can build them into HTML docs. It'd be cool if RTD could run
# sphinx-apidoc for me, since it's easy to forget to regen API docs
# and commit them to my repo after making changes to my code.

PACKAGES = ["asteroid"]


def run_apidoc(_):
    os.makedirs(os.path.join(PATH_HERE, "apidoc"), exist_ok=True)
    for pkg in PACKAGES:
        argv = [
            "-e",
            "-o",
            os.path.join(PATH_HERE, "apidoc"),
            os.path.join(PATH_HERE, PATH_ROOT, pkg),
            "**/test_*",
            "--force",
            "--private",
            "--module-first",
        ]
        try:
            # Sphinx 1.7+
            from sphinx.ext import apidoc

            apidoc.main(argv)
        except ImportError:
            # Sphinx 1.6 (and earlier)
            from sphinx import apidoc

            argv.insert(0, apidoc.__file__)
            apidoc.main(argv)


def setup(app):
    app.connect("builder-inited", run_apidoc)


# copy all notebooks to local folder #FIXME : temp fix
# path_nbs = os.path.join(PATH_HERE, 'notebooks')
# if not os.path.isdir(path_nbs):
#     os.mkdir(path_nbs)
# for path_ipynb in glob.glob(os.path.join(PATH_ROOT, 'notebooks', '*.ipynb')):
#     path_ipynb2 = os.path.join(path_nbs, os.path.basename(path_ipynb))
#     shutil.copy(path_ipynb, path_ipynb2)

# Ignoring Third-party packages
# https://stackoverflow.com/questions/15889621/sphinx-how-to-exclude-imports-in-automodule

MOCK_REQUIRE_PACKAGES = []
with open(os.path.join(PATH_ROOT, "requirements.txt"), "r") as fp:
    for ln in fp.readlines():
        found = [ln.index(ch) for ch in list(",=<>#") if ch in ln]
        pkg = ln[: min(found)] if found else ln
        if pkg.rstrip():
            MOCK_REQUIRE_PACKAGES.append(pkg.rstrip())

# TODO: better parse from package since the import name and package name may differ
MOCK_MANUAL_PACKAGES = ["torch", "torchvision"]
autodoc_mock_imports = MOCK_REQUIRE_PACKAGES + MOCK_MANUAL_PACKAGES
# for mod_name in MOCK_REQUIRE_PACKAGES:
#     sys.modules[mod_name] = mock.Mock()


# Options for the linkcode extension
# ----------------------------------
# github_user = 'mpariente'
# github_repo = 'asteroid'
#
#
# # Resolve function
# # This function is used to populate the (source) links in the API
# def linkcode_resolve(domain, info):
#     def find_source():
#         # try to find the file and line number, based on code from numpy:
#         # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
#         obj = sys.modules[info['module']]
#         for part in info['fullname'].split('.'):
#             obj = getattr(obj, part)
#         fname = inspect.getsourcefile(obj)
#         # https://github.com/rtfd/readthedocs.org/issues/5735
#         if any([s in fname for s in ('readthedocs', 'rtfd', 'checkouts')]):
#             # /home/docs/checkouts/readthedocs.org/user_builds/pytorch_lightning/checkouts/
#             #  devel/pytorch_lightning/utilities/cls_experiment.py#L26-L176
#             path_top = os.path.abspath(os.path.join('..', '..', '..'))
#             fname = os.path.relpath(fname, start=path_top)
#         else:
#             # Local build, imitate master
#             fname = 'master/' + os.path.relpath(fname, start=os.path.abspath('..'))
#         source, lineno = inspect.getsourcelines(obj)
#         return fname, lineno, lineno + len(source) - 1
#
#     if domain != 'py' or not info['module']:
#         return None
#     try:
#         filename = '%s#L%d-L%d' % find_source()
#     except Exception:
#         filename = info['module'].replace('.', '/') + '.py'
#     # import subprocess
#     # tag = subprocess.Popen(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE,
#     #                        universal_newlines=True).communicate()[0][:-1]
#     branch = filename.split('/')[0]
#     # do mapping from latest tags to master
#     branch = {'latest': 'master', 'stable': 'master'}.get(branch, branch)
#     filename = '/'.join([branch] + filename.split('/')[1:])
#     return "https://github.com/%s/%s/blob/%s" \
#            % (github_user, github_repo, filename)


# Autodoc config
autodoc_inherit_docstring = False
autodoc_default_flags = ["members", "show-inheritance"]
# Order functions by appearance in source (default 'alphabetical')
autodoc_member_order = "groupwise"


# autodoc_member_order = 'groupwise'
# # autoclass_content = 'both'
# # autodoc_default_flags = [
# #     'members', 'undoc-members', 'show-inheritance', 'private-members',
# #     # 'special-members', 'inherited-members'
# # ]
# autodoc_default_flags = ['members', 'show-inheritance']
#
# # Autodoc config
# autodoc_inherit_docstring = True
# # autodoc_default_flags = ['members', 'show-inheritance']
# # Order functions by appearance in source (default 'alphabetical')
# # autodoc_member_order = 'bysource'
