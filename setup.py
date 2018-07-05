"""
Example of how to use setuptools
"""

from sift import __version__                    # 1) 从 submodule1.__init__.__version__ 获得版本号

from setuptools import setup, find_packages

# 命令行："python setup.py --version" 可以获得版本号。

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# 打包前执行 `git tag -a $(python setup.py --version)`　将 __version__ 注册为 tag number
setup(
    author='Jeff Wang',
    author_email='jeffwji@test.com',

    version=__version__,

    #version_command='git describe --always --long --dirty=-dev',  # 3) 获得　tag 动态获得版本号(参考文档 <git release flow>)
    # `--always` 如果没有打过标签会出现错误信息 `fatal: No names found, cannot describe anything.`，这个参数将返回 commit hash number 代替 tag 以避免错误.
    # `--long --dirty=-dev` 获得长格式版本信息： <version>-<times>-<commit-hash>-<dirty> 例如：0.0.2-0-g00bd0b4-dev

    name = "sift",
    packages = find_packages(
        exclude=['tests', '*.tests', '*.tests.*']
    ),

    ########
    # 打包规则
    #
    # 定义 src 目录下的子包的打包规则，缺省 setup.py 只打包 py 文件，如果希望加入其他文件，需要在 package_data 中定义。
    package_data = {
        # 任何包中含有 .properties 文件，都包含它
        '':[ 'images/*', '*.md', 'requirements.txt' ],
    },
    #include_package_data=True,
    # MANIFEST.in 文件用于定义其他不存在于 `package_data`(包含 __init__.py ) 范围内的文件。

    install_requires=requirements,
)
