from setuptools import setup


with open('requirements.txt') as f:
    packages = f.read().splitlines()

setup(
    name='ocrmodel',
    packages=['ocr'],
    install_requires=packages
)
