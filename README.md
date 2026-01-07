<div align="center">

# EvoLUT - Evolving 3D Look-Up Tables

[![English](https://img.shields.io/badge/language-English-blue)](README.md)
[![中文](https://img.shields.io/badge/language-中文-red)](README-zh.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

</div>

## 📑 Table of Contents | 目录
- [✨ Overview | 项目简介](#-overview)
- [🚀 Quick Start | 快速开始](#-quick-start)
- [📦 Installation | 安装](#-installation)
- [🎯 Features | 功能特性](#-features)
- [📚 Documentation | 文档](#-documentation)
- [🤝 Contributing | 贡献指南](#-contributing)

## ✨ Overview
With the emergence of large-scale datasets and continuous advancements in computational power, deep learning has achieved significant breakthroughs in computer vision. However, in scenarios with limited training samples, model performance and generalization capabilities remain significantly constrained. Existing few-shot learning methods based on image data augmentation can partially alleviate data scarcity issues, but they often fail to fully exploit the latent information within limited samples, thereby limiting the effective enhancement of data diversity. To overcome this limitation, this study proposes an innovative image data augmentation framework based on a three-dimensional mapping table. This framework first leverages a pre-trained visual model on the original dataset, then employs a co-evolutionary algorithm to perform pixel-level optimization on the 3D Look-Up Table, significantly enhancing sample diversity. Secondly, a pruning strategy is introduced to remove redundant elements in the 3D Look-Up Table that have minimal impact on mapping results, thereby reducing encoding length and computational complexity. The optimized 3D Look-Up Table demonstrates strong generalization capabilities, effectively enhancing unseen data. This design ensures augmented data maintains semantic consistency while exhibiting richer, more informative feature distributions, thereby mitigating the impact of data scarcity. Extensive experiments across multiple benchmark datasets confirm the proposed framework significantly outperforms existing image data augmentation methods on multiple evaluation metrics, validating its effectiveness and superiority.

### 🔬 Core Innovations
### 1. **Evolutionary 3D LUTs Optimization Framework**
We propose a novel evolutionary algorithm-based optimization framework that adaptively searches and constructs 3D Look-Up Table parameters. This approach achieves data augmentation with both **semantic rationality** and **visual diversity** through intelligent parameter optimization.

### 2. **Hilbert Curve Structural Encoding**
By utilizing the spatial filling properties of Hilbert curves, we transform the high-dimensional structure of 3D LUTs into compact one-dimensional sequence encoding. This innovative encoding method **preserves spatial locality** while significantly improving **search efficiency**.

### 3. **Adaptive Encoding Pruning Strategy**
Through comprehensive analysis of gene loci contributions to mapping results, we introduce an intelligent pruning mechanism that eliminates less impactful dimensions. This strategy **reduces encoding length by 30-50%** and substantially **decreases computational overhead** during evolutionary iterations.

### 4. **Intelligent Sampling and Filtering Mechanism**
We establish a robust sampling and selection pipeline that optimizes 3D LUTs on training subsets and filters generated samples based on **visual quality** and **semantic consistency** criteria. This ensures the production of **high-quality augmented data** with proven effectiveness.

## 🚀 Quick Start
## Clone the project locally
git clone https://github.com/ouyangyang258/EvoLUT.git

cd EvoLUT
## Install dependencies (recommended to use a virtual environment)
pip install -r requirements.txt

## 📦 Installation

## 🎯 Features

## 📖 Usage Examples

## 📚 Documentation

## 🤝 Contributing

Thank you for your interest in the EvoLUT project! Guidelines and instructions for contributing will be provided next.

### Quick Start
1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/EvoLUT.git`
3. **Install** dev dependencies: `pip install -e ".[dev]"`
4. **Create** a branch: `git checkout -b feature/your-feature`
5. **Make** your changes and test: `pytest`
6. **Commit**: `git commit -m "feat: add new feature"`
7. **Push**: `git push origin feature/your-feature`
8. **Open** a Pull Request

### Guidelines
- Follow [PEP 8](https://pep8.org/) style
- Add tests for new features
- Update documentation
- Use descriptive commit messages

### Need Help?
- Ask in GitHub Discussions
- Review existing PRs for examples

Every contribution, big or small, is appreciated! 🎉

## 🙏 Acknowledgments
- We extend our gratitude to all contributors for their invaluable contributions.
- We thank the open-source community for its support.
- We acknowledge the inspiration drawn from relevant research.
- We appreciate users for their usage and feedback.