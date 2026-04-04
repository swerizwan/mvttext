# MvtText: A Multistage Latent Diffusion Framework for Text-Driven Human Motion Generation

<img style="max-width: 100%;" src="https://github.com/swerizwan/mvttext/blob/main/resources/wax.png" alt="Title Overview">

## Overview

MvtText is a hierarchical framework for generating realistic human motion from natural language descriptions. It combines Variational Autoencoders with a cascaded latent diffusion model to progressively refine motion representations across multiple abstraction levels. A dynamic fusion mechanism ensures tight alignment between text and motion, while a novel segment-aware module enhances semantic consistency. Experiments on benchmark datasets show that MvtText achieves superior performance in accuracy, diversity, and realism, setting a new state of the art in text-to-motion generation.

# 👁️💬 Architecture

## 🧾 Framework Overview

Overview of the proposed MvtText framework. Given a textual prompt, the text encoder extracts a semantic embedding of dimension \\(d\\), while the multi-level pose VAE encodes hierarchical movement embeddings \\((y_1, y_2, y_3, y_4)\\). A Latent Diffusion Decoder refines these representations, and the Dynamic Multi-Condition Fusion module adaptively integrates text and motion information at each abstraction level.

<img style="max-width: 100%;" src="https://github.com/swerizwan/mvttext/blob/main/resources/Fig2.png" alt="VERHM Overview">

# Installation

```
conda create python=3.9 --name MvtText
conda activate MvtText
```
Install the requirements
```
pip install -r requirements.txt
```

# Demo

```
python demo.py --cfg ./configs/config_motiontext_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example demo.txt

/home/abbas/motiontext/blender/blender-2.83.0-linux64/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=/home/abbas/motiontext/outcomes/motiontext/HumanML3D/samples_2024-11-10-18-50-15/ --mode=video --joint_type=HumanML3D

python -m fit --dir /home/abbas/motiontext/outcomes/motiontext/HumanML3D/samples_2024-11-10-18-50-15/ --save_folder /home/abbas/motiontext/outcomes/motiontext/HumanML3D/samples_2024-11-10-18-50-15/tamp --cuda True

/home/abbas/motiontext/blender/blender-2.83.0-linux64/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=/home/abbas/motiontext/results/motiontext/1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/samples_2024-10-18-22-15-14/ --mode=video --joint_type=HumanML3D
```

Qualitative results demonstrating MvtText's capability to synthesize human movement for body language synthesis from textual descriptions.

<table>
  <tr>
    <td style="text-align: center;">
      <p>A person kicks with left leg.</p>
      <img width="165" src="https://github.com/swerizwan/MvtText/blob/main/resources/kicking.gif" alt="Happy">
    </td>
    <td style="text-align: center;">
      <p>A person dropped to left knees.</p>
      <img width="165" src="https://github.com/swerizwan/MvtText/blob/main/resources/kneeleft.gif" alt="Frustrated">
    </td>
    <td style="text-align: center;">
      <p>A person is diving into the pool.</p>
      <img width="165" src="https://github.com/swerizwan/MvtText/blob/main/resources/diving.gif" alt="Sad">
    </td>
    <td style="text-align: center;">
      <p>A person is dancing joyfully.</p>
      <img width="165" src="https://github.com/swerizwan/MvtText/blob/main/resources/dancing.gif" alt="Angry">
    </td>
  </tr>
    <tr>
    <td style="text-align: center;">
      <p>A person raises his right hand and waves.</p>
      <img width="165" src="https://github.com/swerizwan/MvtText/blob/main/resources/waves.gif" alt="Happy">
    </td>
    <td style="text-align: center;">
      <p>A person jumps with both feet together.</p>
      <img width="165" src="https://github.com/swerizwan/MvtText/blob/main/resources/jumped.gif" alt="Frustrated">
    </td>
    <td style="text-align: center;">
      <p>A person stretches his arms above his head.</p>
      <img width="165" src="https://github.com/swerizwan/MvtText/blob/main/resources/stretches.gif" alt="Sad">
    </td>
    <td style="text-align: center;">
      <p>A person kneels down on his right knee.</p>
      <img width="165" src="https://github.com/swerizwan/MvtText/blob/main/resources/kneeright.gif" alt="Angry">
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <p>A person walks forward with waving arms.</p>
      <img width="165" src="https://github.com/swerizwan/MvtText/blob/main/resources/walking.gif" alt="Happy">
    </td>
    <td style="text-align: center;">
      <p>A person crawls around in a circle.</p>
      <img width="165" src="https://github.com/swerizwan/MvtText/blob/main/resources/circle.gif" alt="Frustrated">
    </td>
    <td style="text-align: center;">
      <p>A person performs a star jump.</p>
      <img width="165" src="https://github.com/swerizwan/MvtText/blob/main/resources/starjump.gif" alt="Sad">
    </td>
    <td style="text-align: center;">
      <p>A person performs a cartwheel.</p>
      <img width="165" src="https://github.com/swerizwan/MvtText/blob/main/resources/cartwheel.gif" alt="Angry">
    </td>
  </tr>
</table>

## Datasets

We evaluated MvtText using three key datasets for text-driven human movement synthesis:

- **HumanML3D**: Combines HumanAct12 and AMASS datasets, featuring 14,616 movements and 44,970 text descriptions. It spans diverse actions like daily tasks, athletics, and performances, with clips totaling 28.59 hours. Each movement has 3-4 descriptive sentences. [Dataset Link](https://cove.thecvf.com/datasets/682) 
- **KIT-ML**: Includes 3,911 movements with 6,278 text descriptions, linking human actions to natural language. It advances research on movement-language correlations with a focus on accessibility and clarity. [Dataset Link](https://motion-annotation.humanoids.kit.edu/dataset/) 
- **HumanAct12**: Contains 1,191 movement clips and 90,099 poses, categorized into 12 action classes and 34 sub-classes. Adapted from PHSPD, it supports detailed movement-text pairing. [Dataset Link](https://service.tib.eu/ldmservice/en/dataset/humanact12) 

## Train & Evaluate

```
python -m train --cfg configs/config_vae_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
python -m train --cfg configs/config_motiontext_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
python -m train --cfg configs/config_motiontext_humanact12.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
python -m train --cfg configs/config_vae_kit.yaml --cfg_assets configs/assets.yaml --batch_size 32 --nodebug
```

## Running the Demo

To run the demo, follow these steps:

1. Download Blender from [Blender Official Website](https://www.blender.org/download/), and place it in the Blender folder within the root directory.
3. Run the demo by executing `python demo.py --cfg ./configs/config_motiontext_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example demo.txt` with the desired input voice. 
