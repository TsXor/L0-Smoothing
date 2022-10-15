import setuptools, pathlib

wheel_dir = pathlib.Path(__file__).parent

with open(wheel_dir/"README.md") as f:
    long_description = f.read()

setuptools.setup(
    name="L0-Smoothing",
    version="0.1.3",
    license='MIT',

    author="Nrupatunga(normal), TsXor(pyocl)",
    author_email="nrupatunga.s@byjus.com, zhang050525@qq.com",

    description="Implementation of 《Image Smoothing via L0 Gradient Minimization》",
    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=setuptools.find_packages(),
    package_data = {
        '': ['*.cl'],
    },

    url="https://github.com/TsXor/L0-Smoothing",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Environment :: GPU",
    ],

    install_requires=[
        "numpy",
        "opencv_python",
        "pyopencl",
        "pyvkfft",
    ],

    entry_points={
        'console_scripts': [
            'L0-Smoothing = L0_Smoothing.cli:main',
        ]
    },

    python_requires=">=3"
)