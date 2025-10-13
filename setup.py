"""
Orion - Video Analysis Pipeline
Transform videos into queryable knowledge graphs
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="orion-research",
    version="0.1.0",
    description="AI-powered video analysis pipeline with knowledge graph generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Riddhiman Rana",
    author_email="riddhimanrana@example.com",
    url="https://github.com/riddhimanrana/orion-research",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "transformers>=4.48.0",
        "accelerate>=1.6.0",
        
        # Computer Vision
        "ultralytics>=8.3.0",
        "opencv-python>=4.7.0",
        "timm>=1.0.15",
        
        # ML/AI
        "sentence-transformers>=3.3.0",
        "hdbscan>=0.8.39",
        "scikit-learn>=1.2.2",
        
        # Knowledge Graph
        "neo4j>=5.26.0",
        
        # API/Web
        "requests>=2.32.0",
        "httpx>=0.28.0",
        
        # CLI/UI
        "rich>=14.1.0",
        "typer>=0.19.0",
        
        # Utilities
        "numpy>=1.26.4",
        "pandas>=2.3.3",
        "pillow>=11.3.0",
        "tqdm>=4.67.0",
        "huggingface-hub>=0.35.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.13.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "orion=orion:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
