# Models for NegBench Evaluation

This directory contains pretrained models for evaluating vision-language models' ability to understand negation. The models are organized as follows:

```
models/
├── ConCLIP/
│   ├── conclip_b32_openclip_version.pt
├── NegCLIP/
│   ├── negclip.pth
├── NegCLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01/
│   ├── finetuned_checkpoint.pt
```

## Available Models

All models in this directory use **ViT-B-32** as the CLIP architecture.

### 1. **ConCLIP** (WACV 2025)
- `conclip_b32_openclip_version.pt`  
- A contrastively trained CLIP model designed to improve negation understanding.
- **[Original paper link](https://arxiv.org/abs/2403.20312)**

### 2. **NegCLIP** (ICLR 2023)
- `negclip.pth`  
- An improved version of CLIP, fine-tuned for better compositional language understanding.
- **[Original paper link](https://openreview.net/forum?id=KRLUvxh8uaX)**

### 3. **NegCLIP_CC12M_NegFull** (CVPR 2025)
- `finetuned_checkpoint.pt`  
- A version of NegCLIP that we fine-tuned on the **CC12M-NegFull** dataset, which combines **CC12M-NegCap** and **CC12M-NegMCQ**. This is our best model in terms of negation understanding capabilities. 

---

## Downloading Models

All models can be downloaded from the following **[Google Drive link](https://drive.google.com/drive/folders/1kSEq0mkV1t1T8GuOAM65iz_iAA7e5gxB?usp=sharing)**.

Once downloaded, place the models in the `models/` directory as shown above.

---

## Using Pretrained Models in Benchmarks

To evaluate one of these models using the scripts in `/benchmarks`, specify the model path in the `PRETRAINED_MODEL` variable.

Example usage:
```bash
MODEL=ViT-B-32
PRETRAINED_MODEL=models/NegCLIP/negclip.pth
```

---

## Using OpenCLIP Models

You do **not** need to download OpenCLIP models manually. Instead, specify the desired architecture and pretraining dataset in the benchmark scripts.

Example:
```bash
MODEL=ViT-B-32
PRETRAINED_MODEL=openai
```
An example with a CLIP pretrained on a different dataset:
```bash
MODEL=ViT-B-32
PRETRAINED_MODEL=laion2b_s34b_b79k
```

For a list of available OpenCLIP models and pretraining datasets, refer to the **[OpenCLIP repository](https://github.com/mlfoundations/open_clip)**.

---

## Contact and Updates

The code supports a lot more models. Stay tuned for updated instructions that covers more models. For any issues or questions, please reach out via GitHub issues or email the authors.