"""
Example of how to use setuptools
"""

from sift import __version__

from setuptools import setup, find_packages

# 命令行："python setup.py --version" 可以获得版本号。

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# 打包前执行 `git tag -a $(python setup.py --version)`　将 __version__ 注册为 tag number
setup(
    author='Jeff Wang',
    author_email='jeffwji@test.com',

    version=__version__,

    name = "sift",
    packages = find_packages(
        exclude=['tests', '*.tests', '*.tests.*']
    ),

    package_data = {
        '':[ 'images/*', '*.md', 'requirements.txt' ],
    },
    #include_package_data=True,
    # MANIFEST.in 文件用于定义其他不存在于 `package_data`(包含 __init__.py ) 范围内的文件。

    install_requires=requirements,
)
