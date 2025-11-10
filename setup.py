from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="self-evolving-teacher-student",
    version="0.1.0",
    author="Nguyen Trung Hieu",
    author_email="hieuhip4444@gmail.com",
    description="A Self-Evolving Teacher-Student Architecture for Scalable and Cost-Efficient LLM Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sidataba/LLMs--Self-Evolving-Teacher-Student-Architecture",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "openai>=1.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "viz": [
            "tensorboard>=2.13.0",
            "plotly>=5.14.0",
            "matplotlib>=3.7.0",
        ],
    },
)
