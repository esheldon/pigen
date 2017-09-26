import distutils
from distutils.core import setup

scripts=['pigen']
scripts=[os.path.join('bin',s) for s in scripts]

setup(
    name="pigen", 
    packages=['pigen'],
    version="0.1.0",
    scripts=scripts,
)



