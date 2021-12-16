from setuptools import setup


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

packages = []
dependencies = []
for package in requirements:
    if package:
        if 'find-links' in package:
            dependencies.append(package.split()[1])
        else:
            packages.append(package)

setup(
    name='ocrmodel',
    packages=['ocr'],
    install_requires=packages,
    dependency_links=dependencies
)
