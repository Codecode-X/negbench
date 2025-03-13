# Vision-Language Models Do *Not* Understand Negation *(CVPR 2025)*

This repository contains the code, datasets, and resources for the paper **"Vision-Language Models Do *Not* Understand Negation"** ([preprint link](https://arxiv.org/abs/2501.09425)) **accepted at CVPR 2025.** The paper explores the limitations of vision-language models (e.g., CLIP, NegCLIP) in understanding negation and presents new evaluation benchmarks and fine-tuning datasets to address these challenges.

This repository is **a work in progress**, and the authors welcome feedback, suggestions, and contributions. We are also happy to discuss extensions of our work. Please feel free to open an issue on GitHub or reach out via email.

---

## Updates & Changelog

- **[03/2025]** 🛠 **Bug Fix:** Added feature normalization (L2 normalization of image and text embeddings) before computing the dot product in MCQ evaluations of CLIP-like models. The code now correctly applies normalization. This change affects the numbers in Figure 4 of the preprint, but the overall trends remain consistent. See updated values in [`results/mcq/1_baseline_total.csv`](results/mcq/1_baseline_total.csv) and [`results/mcq/2_scaling_clip_total.csv`](results/mcq/2_scaling_clip_total.csv). Our updated finetuned model numbers are in [`results/mcq/3_finetuned_total.csv`](results/mcq/3_finetuned_total.csv).
- **[03/2025]** 🎉 **NegBench is accepted to CVPR 2025!**
- **[01/2025]** 🚀 **Initial Release:** Code to reproduce main retrieval and MCQ results of the preprint is now available.

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

### Supported Models
For instructions and links to download some of the evaluated models (e.g., OpenAI CLIP, CoNCLIP, NegCLIP, our finetuned NegCLIP), refer to [`models.md`](models.md).

---

## Feedback and Support
We value feedback from the community! If you have questions, comments, or suggestions, feel free to:
- Open an issue on this repository.
- Email the authors directly.

---

## Citation
If you find this work useful in your research, please cite our paper:

```bibtex
@article{alhamoud2025vision,
  title={Vision-Language Models Do Not Understand Negation},
  author={Alhamoud, Kumail and Alshammari, Shaden and Tian, Yonglong and Li, Guohao and Torr, Philip and Kim, Yoon and Ghassemi, Marzyeh},
  journal={arXiv preprint arXiv:2501.09425},
  year={2025}
}
```

Thank you for your interest in this project! We look forward to your feedback and collaboration.

## Contact

For questions or feedback, please reach out to:

- **Kumail Alhamoud**: [kumail@mit.edu](mailto:kumail@mit.edu)
- **Shaden Alshammari**: [shaden@mit.edu](mailto:shaden@mit.edu)
