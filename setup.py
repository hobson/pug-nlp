# setup.py for PUG (PDX Python User Group) package
# the parent name (perhaps a namespace package) you'd import
__namespace_package__ = 'pug'
# the subpackage that this installer is providing that you'd import like __import__(__namespace_package__ + '.' + '__subpackage__')
__subpackage__ = 'nlp'
# the name as it will appear in the pypi cheeseshop repositor, not the name you'd use to import it
project_name = '{}-{}'.format(__namespace_package__, __subpackage__)
package_name = '{}.{}'.format(__namespace_package__, __subpackage__)

from setuptools import find_packages
from distutils.core import setup
import os

# # If you want tests to work with django settings.py you need django-setuptest
# from setuptest import test
# # If you want to use setuptest.test instead of the python test,
# #    you need to say so in your setup(kwargs) below, like this:
# # setup(cmdclass={'test': test},...

print('Installing package named {} from the {} project, a sub-package/project of the namespace package {}. . .'.format(package_name, project_name, package_name))

global_env, env = {}, {}
execfile(os.path.join(__namespace_package__, __subpackage__, 'package_info.py'), global_env, env)

print('Found package info: {}'.format(env))

version = env.get('__version__', '0.0.1')
package_docstring = env.get('__doc__', '`{}` python package'.format(project_name))
description = package_docstring.split('\n')[0]
long_description = package_docstring
__url__  = env.get('__url__', 'http://github.com/hobson/')
__authors__  = env.get('__authors__', ('Hobson <hobson@totalgood.com>',))
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst', 'md')
except:  # (IOError, ImportError, OSError, RuntimeError):
    # from traceback import print_exc
    # print_exc()
    print('Unable to use pypandoc to reformat the README.md file into RST format')

print('Installing package named {} from the {} project. . .'.format(package_name, project_name))

try:
    import pip
    # print(pip.__version__)
    import uuid
    # print(uuid.uuid1)
    from pip.req import parse_requirements
    requirements = list(parse_requirements('requirements.txt', session=uuid.uuid1()))
except:
    from traceback import print_exc
    print_exc()
    requirements = []
print('requirements: {}'.format(requirements))

install_requires=[str(req.req).split(' ')[0].strip() for req in requirements if req.req and not req.url]
print('Install requires: {}'.format(install_requires))
dependency_links=[req.url for req in requirements if req.url]
print('Dependency links: {}'.format(dependency_links))


EXCLUDE_FROM_PACKAGES = []

setup(
    name=project_name,
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    namespace_packages=[__namespace_package__],

    # install non-.py files listed in MANIFEST.in (.js, .html, .txt, .md, etc)
    include_package_data = True,
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
    download_url = "{}/tarball/master".format(__url__),
    keywords = ["nlp", "natural language processing", "text", "text processing", "bot", "ai", "agent", "data", "science", "data science", "math", "machine-learning", "statistics", "database"],
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
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        ],
)
