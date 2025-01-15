This directory contains all the scripts and resources required to construct synthetic datasets for evaluating and fine-tuning vision-language models' understanding of negation. The datasets align with the methodology outlined in the paper and are organized into two subdirectories:

---

## Directory Structure

### `evaluation/`
- **Description**: Contains scripts for generating evaluation datasets (e.g., NegBench) to assess vision-language models' ability to understand negation.
- **Corresponds to**: Section 4 of the paper.
- **Outputs**:
  - NegBench datasets, including synthetic captions and multiple-choice questions (MCQs), designed to evaluate models' negation understanding across diverse linguistic templates.
- **Datasets**:
  - Relies on **COCO**, **VOC2007**, and **MSR-VTT** datasets as inputs. 
  - Refer to `docs/datasets.md` for instructions on how to download and prepare these datasets.
- **Key Steps**:
  1. Process captions to extract positive and negative objects.
  2. Verify the absence of negative objects.
  3. Generate MCQs using templates.
  4. Paraphrase captions to enhance linguistic diversity.

### `finetuning/`
- **Description**: Contains scripts for creating fine-tuning datasets with negated captions and MCQs.
- **Corresponds to**: Section 5 of the paper.
- **Outputs**:
  - **CC12M-NegCap**: Augmented captions with negated objects (~30 million captions).
  - **CC12M-NegMCQ**: Four captions per image (1 correct + 3 hard negatives) for fine-tuning (~40 million captions).
  - **CC12M-NegFull**: A combination of CC12M-NegCap and CC12M-NegMCQ.
- **Datasets**:
  - Relies on the **CC12M** dataset as the base input.
  - Ensure that you have access to the CC12M dataset for processing.
- **Key Steps**:
  1. Extract positive and negative objects.
  2. Verify object annotations.
  3. Generate negated captions.
  4. Create and paraphrase MCQs.

---

## General Notes
- **Requirements**: Both subdirectories use the same Conda environment for dependency management.
- **Execution**: Example bash scripts are provided to demonstrate distributed processing across GPU nodes.
- **Customization**: Update bash script placeholders with environment-specific paths and tokens.
- **Datasets**: Refer to `docs/datasets.md` for detailed instructions on downloading and preparing the required datasets.

Refer to the respective `README.md` files in each subdirectory for detailed instructions and step-by-step guides.
