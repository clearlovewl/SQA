<!--
 * @Author: ll
 * @LastEditTime: 2024-06-18 12:53:25
 * @LastEditors: ll
 * 无问西东
-->
<div align="center">
<!-- DO NOT ADD CONDA DOWNLOADS... README CHANGES MUST BE APPROVED BY EDEN OR WILL -->

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)](https://pypi.org/project/pytorch-lightning/)
[![PyPI Status](https://badge.fury.io/py/pytorch-lightning.svg)](https://badge.fury.io/py/pytorch-lightning)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pytorch-lightning)](https://pepy.tech/project/pytorch-lightning)
[![Conda](https://img.shields.io/conda/v/conda-forge/lightning?label=conda&color=success)](https://anaconda.org/conda-forge/lightning)
[![codecov](https://codecov.io/gh/Lightning-AI/pytorch-lightning/graph/badge.svg?token=SmzX8mnKlA)](https://codecov.io/gh/Lightning-AI/pytorch-lightning)

[![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/lightning-ai/lightning)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning/blob/master/LICENSE)

<!--
[![CodeFactor](https://www.codefactor.io/repository/github/Lightning-AI/lightning/badge)](https://www.codefactor.io/repository/github/Lightning-AI/lightning)
-->


</div>

<div align="center">
  
<p align="center">

&nbsp;
  
<a target="_blank" href="https://lightning.ai/docs/pytorch/latest/starter/introduction.html#define-a-lightningmodule">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>

</p>

</div>

# SQA (Subjective bias Quality Assessment)
Implementation of <a href="https://arxiv.org/abs/2311.15846">Learning with Noisy Low-Cost MOS for Image Quality Assessment via Dual-Bias Calibration</a>, in Pytorch.
<!-- <img src="./titok.png" width="400px"></img> -->


## Install

```bash
lightning 
pytorch
```


## Core code

MODEL/LCMOS/model.py

## Usage

```bash
python IQA.py
```
There are many function in IQA.py coresponding to different experiments.
## Citations

```bibtex
@article{wang2023learning,
  title={Learning with Noisy Low-Cost MOS for Image Quality Assessment via Dual-Bias Calibration},
  author={Wang, Lei and Wu, Qingbo and Yuan, Desen and Ngan, King Ngi and Li, Hongliang and Meng, Fanman and Xu, Linfeng},
  journal={arXiv preprint arXiv:2311.15846},
  year={2023}
}
```