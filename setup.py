# setup.py for PUG (PDX Python User Group) package
__namespace_package__ = 'pug'
__subpackage__ = 'ann'
package_name = '{}-{}'.format(__namespace_package__, __subpackage__)

from setuptools import find_packages
from distutils.core import setup
import os

# # If you want tests to work with django settings.py you need django-setuptest
# from setuptest import test
# # If you want to use setuptest.test instead of the python test,
# #    you need to say so in your setup(kwargs) below, like this:
# # setup(cmdclass={'test': test},...

global_env, env = {}, {}
execfile(os.path.join(__namespace_package__, package_name, 'package_info.py'), global_env, env)

version = env.get('__version__', '0.0.1')
long_description = env.get('__doc__', '0.0.1')
package_docstring = env.get('__doc__', '`{}` python package'.format(package_name))
description = package_docstring.split(',')[0]
long_description = package_docstring
__url__  = env.get('__url__', 'http://github.com/hobson/')
__authors__  = env.get('__authors__', ('Hobson <hobson@totalgood.com>',))
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError, OSError):
    pass

print('Installing package named {}. . .'.format(package_name))

try:
    from pip.req import parse_requirements
    requirements = list(parse_requirements('requirements.txt'))
except:
    requirements = []
install_requires=[str(req).split(' ')[0].strip() for req in requirements if req.req and not req.url]
print('Install requires: {}'.format(install_requires))
dependency_links=[req.url for req in requirements if req.url]
print('Dependency links: {}'.format(dependency_links))


EXCLUDE_FROM_PACKAGES = []

setup(
    name = package_name,
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),   #[package_name],  
    include_package_data = True,  # install non-.py files listed in MANIFEST.in (.js, .html, .txt, .md, etc)
    install_requires = install_requires,
    dependency_links = dependency_links,
    # scripts=['pug/bin/test_ann.py'],
    # entry_points={'console_scripts': [
    #     'test-ann = pug.ann.tests.run',
    # ]},
    version = version,
    description = description,
    long_description = long_description,
    author = ', '.join(__authors__),
    author_email = __authors__[0].split('<')[1].strip().strip('>'),

    #tests_require = ['django-setuptest', 'south'],
    #test_suite = 'setuptest.setuptest.SetupTestSuite',
    #cmdclass = {'test': test},
    url = __url__,

    # Force setup.py to use the latest github master source files rather than the cheeseshop tarball: 
    download_url = "{0}/tarball/master".format(__url__),
    keywords = ["agent", "bot", "ai", "crawl", "data", "science", "data science", "math", "machine-learning", "statistics", "database"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        # "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        ],
)