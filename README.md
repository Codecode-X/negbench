# Vision-Language Models Do Not Understand Negation

This repository contains the code, datasets, and resources for the paper **"Vision-Language Models Do Not Understand Negation"** ([preprint link](#)). The paper explores the limitations of vision-language models (e.g., CLIP, NegCLIP) in understanding negation and presents new evaluation benchmarks and fine-tuning datasets to address these challenges.

This repository is **a work in progress**, and the authors welcome feedback, suggestions, and contributions. Please feel free to open an issue on GitHub or reach out via email.

---

## Repository Structure

The repository is organized as follows:

### 1. `benchmarks/`
- Contains a comprehensive benchmark (NegBench) for evaluating vision-language models on negation-specific tasks.

### 2. `synthetic_datasets/`
- Scripts for constructing evaluation and fine-tuning datasets with negation-specific examples.
- Subdirectories:
  - `evaluation/`: Tools for creating datasets to evaluate negation understanding (e.g., NegBench).
  - `finetuning/`: Tools for creating datasets to fine-tune models on negation tasks (e.g., CC12M-NegCap, CC12M-NegMCQ).

Each subdirectory contains its own `README.md` file with detailed instructions on how to use the scripts and files.

---

### Dataset Preparation
For detailed instructions on downloading and preparing datasets (e.g., CC12M, COCO, VOC2007, MSR-VTT), refer to [`datasets.md`](datasets.md).

---

## Feedback and Support
We value feedback from the community! If you have questions, comments, or suggestions, feel free to:
- Open an issue on this repository.
- Email the authors directly.

---

## Citation
If you find this work useful in your research, please cite our paper:

```bibtex
@article{negbench2024,
  title={Vision-Language Models Do Not Understand Negation},
  author={Author Name and Co-author Name},
  journal={Preprint},
  year={2024},
  url={https://arxiv.org/abs/XXXX.XXXX}
}
```

Thank you for your interest in this project! We look forward to your feedback and collaboration.

## Contact

For questions or feedback, please reach out to:

- **Kumail Alhamoud**: [kumail@mit.edu](mailto:kumail@mit.edu)
- **Shaden Alshammari**: [shaden@mit.edu](mailto:shaden@mit.edu)