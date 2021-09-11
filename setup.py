from setuptools import setup

setup(
    name="CEC2017",
    version="1.0",
    description="CEC 2017 single objective optimization suite",
    author="Duncan Tilley",
    url="https://github.com/tilleyd/cec2017-py",
    packages=["cec2017"],
    include_package_data=True,
    install_requires=["numpy"],
)
