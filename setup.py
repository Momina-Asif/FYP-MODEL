"""
Setup configuration for Poster Generation Framework
"""

from setuptools import setup, find_packages

with open("README_FRAMEWORK.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="poster-generator",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered poster generation using Stable Diffusion and custom models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/poster-generator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "Pillow>=9.5.0",
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "xformers>=0.0.22",
        ],
    },
    entry_points={
        "console_scripts": [
            "poster-generate=cli:main",
        ],
    },
)
