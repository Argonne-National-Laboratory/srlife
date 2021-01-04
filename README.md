# srlife: solar receiver life estimation tool

![Test Status](https://github.com/Argonne-National-Laboratory/srlife/workflows/tests/badge.svg?branch=master) ![Documentation Status](https://readthedocs.org/projects/srlife/badge/?version=latest)

This python package evaluates the structural life of tubular solar receivers against
creep rupture, fatigue, and creep-fatigue damage modes.

Full documentation is available [here](https://srlife.readthedocs.io/).

## Prerequisites

The package itself is pure python, however it relies on several additional
python packages listed in [requirements.txt](requirements.txt).
Note that srlife uses Python3.

Of these additional requirements, only [neml](https://github.com/Argonne-National-Laboratory/neml)
is difficult to install as currently the maintainers do not provide binary packages.
See the installation instructions in the full documentation for how to install srlife and dependencies.

## License

The package is provided under an MIT license found in the
[LICENSE](LICENSE) file.
