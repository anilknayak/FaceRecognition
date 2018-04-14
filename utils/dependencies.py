import pip
import pkg_resources

pkgs = ['numpy', 'tqdm', 'cv2','pyqtgraph','dlib','tensorflow','scikit-learn','protobuf','matplotlib','keras','imutils']
installed_pkgs = [pkg.key for pkg in pip.get_installed_distributions()]



# conda install -c conda-forge keras
# conda install -c menpo opencv
# conda install -c conda-forge pyqtgraph
# conda install -c conda-forge tqdm
# pip install tensorflow
# conda install -c anaconda protobuf
# conda install -c stuwilkins protobuf
# conda install -c conda-forge matplotlib
# conda install -c anaconda keras-gpu

# conda install -c mlgill imutils
# conda install -c anaconda pyzmq
# conda install -c conda-forge typing
# conda install -c conda-forge decorator
# conda install -c conda-forge pytables
# conda install -c conda-forge packaging
# conda install -c conda-forge jupyterlab_launcher

# conda install -c conda-forge dlib=19.4
# conda install -c conda-forge pillow
# conda install -c numba numba


# conda create --name tf --clone root

#
# for pkg in pkgs:
#     if pkg in installed_pkgs:
#         # print('Present')
#         print(pkg, pkg_resources.get_distribution(pkg).version)
#     else:
#         pip.main(['install', pkg])
#         # print(pkg, ' **** NOT **** Present')

print(installed_pkgs)
print('not')

for pkg in pkgs:
    if pkg in installed_pkgs:
        print(pkg, 'present')
    else:
        print(pkg , "not")

['zict', 'xlwt', 'xlsxwriter', 'xlrd', 'wrapt', 'widgetsnbextension', 'wheel', 'werkzeug', 'webencodings', 'wcwidth', 'urllib3', 'unicodecsv', 'typing', 'traitlets', 'tqdm', 'tornado', 'toolz', 'theano', 'testpath', 'terminado', 'termcolor', 'tensorflow-tensorboard', 'tensorflow-gpu', 'tensorboard', 'tblib', 'tables', 'sympy', 'statsmodels', 'sqlalchemy', 'spyder', 'sphinxcontrib-websupport', 'sphinx', 'sortedcontainers', 'sortedcollections', 'snowballstemmer', 'six', 'singledispatch', 'simplegeneric', 'setuptools', 'seaborn', 'scipy', 'scikit-learn', 'scikit-image', 'ruamel-yaml', 'rope', 'requests', 'qtpy', 'qtconsole', 'qtawesome', 'pyzmq', 'pyyaml', 'pywavelets', 'pytz', 'python-dateutil', 'pytest', 'pysocks', 'pyparsing', 'pyopenssl', 'pyodbc', 'pylint', 'pygpu', 'pygments', 'pyflakes', 'pycurl', 'pycrypto', 'pycparser', 'pycosat', 'pycodestyle', 'py', 'ptyprocess', 'psutil', 'protobuf', 'prompt-toolkit', 'ply', 'pkginfo', 'pip', 'pillow', 'pickleshare', 'pexpect', 'pep8', 'patsy', 'pathlib2', 'path.py', 'partd', 'pandocfilters', 'pandas', 'packaging', 'openpyxl', 'opencv-python', 'olefile', 'odo', 'numpydoc', 'numpy', 'numexpr', 'numba', 'notebook', 'nose', 'nltk', 'networkx', 'nbformat', 'nbconvert', 'navigator-updater', 'multipledispatch', 'msgpack-python', 'mpmath', 'mistune', 'mccabe', 'matplotlib', 'markupsafe', 'markdown', 'lxml', 'locket', 'llvmlite', 'lazy-object-proxy', 'keras', 'jupyterlab', 'jupyterlab-launcher', 'jupyter-core', 'jupyter-console', 'jupyter-client', 'jsonschema', 'jinja2', 'jedi', 'jdcal', 'itsdangerous', 'isort', 'ipywidgets', 'ipython', 'ipython-genutils', 'ipykernel', 'imutils', 'imgaug', 'imagesize', 'imageio', 'idna', 'html5lib', 'heapdict', 'h5py', 'grpcio', 'greenlet', 'gmpy2', 'glob2', 'gevent', 'gast', 'flask', 'flask-cors', 'filelock', 'fastcache', 'et-xmlfile', 'entrypoints', 'docutils', 'distributed', 'decorator', 'datashape', 'dask', 'cytoolz', 'cython', 'cycler', 'cryptography', 'contextlib2', 'conda', 'conda-verify', 'conda-build', 'colorama', 'clyent', 'cloudpickle', 'click', 'chardet', 'cffi', 'certifi', 'bottleneck', 'boto', 'bokeh', 'bleach', 'blaze', 'bkcharts', 'bitarray', 'beautifulsoup4', 'backports.shutil-get-terminal-size', 'backports.functools-lru-cache', 'babel', 'astropy', 'astroid', 'astor', 'asn1crypto', 'anaconda-project', 'anaconda-navigator', 'anaconda-client', 'alabaster', 'absl-py', 'mako', 'object-detection']