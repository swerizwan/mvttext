# MvtText: AutoEncoder and Latent Diffusion for Text-Driven Body Language Synthesis

<img src="https://camo.githubusercontent.com/2722992d519a722218f896d5f5231d49f337aaff4514e78bd59ac935334e916a/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67" alt="Oryx Video-ChatGPT" data-canonical-src="https://i.imgur.com/waxVImv.png" style="max-width: 100%;">

## Overview

This paper introduces MvtText, a method combining VAEs and latent diffusion models to generate realistic human movements from text. Using a multi-stage process, it progressively refines movements while aligning them with text inputs through dynamic multi-condition fusion. Experiments show MvtText outperforms existing methods, with applications in animation, VR, and human-computer interaction.

# 👁️💬 Architecture

The MvtText framework works as follows: (a) Movement data is modeled dynamically, and text descriptions are encoded using CLIP. (b) Encoder-decoder pairs generate low-dimensional pose representations. (c) A cascaded latent diffusion process refines these representations iteratively, starting coarse and adding details. (d) The result is realistic 3D human movement sequences aligned with the input text. 

<img style="max-width: 100%;" src="https://github.com/swerizwan/MvtText/blob/main/resources/Fig2.png" alt="VERHM Overview">

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

- **HumanML3D**: Combines HumanAct12 and AMASS datasets, featuring 14,616 movements and 44,970 text descriptions. It spans diverse actions like daily tasks, athletics, and performances, with clips totaling 28.59 hours. Each movement has 3-4 descriptive sentences. [Dataset Link](https://drive.google.com/file/d/1rmnG-R8wTb1sRs0PYp4RRmLg8XH-qSGW/view) 
- **KIT-ML**: Includes 3,911 movements with 6,278 text descriptions, linking human actions to natural language. It advances research on movement-language correlations with a focus on accessibility and clarity. [Dataset Link](https://drive.google.com/file/d/1IXRBm4qSjLQxp1J3cqv1xd8yb-RQY0Jz/view) 
- **HumanAct12**: Contains 1,191 movement clips and 90,099 poses, categorized into 12 action classes and 34 sub-classes. Adapted from PHSPD, it supports detailed movement-text pairing. [Dataset Link](https://drive.google.com/drive/folders/1TBY2x-gD6f3yzQ0WNmXP2-be3xu3qDkV?usp=sharing) 

# Train & Evaluate

- **Train**
```
python -m train --cfg configs/config_vae_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
python -m train --cfg configs/config_motiontext_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
python -m train --cfg configs/config_motiontext_humanact12.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
python -m train --cfg configs/config_vae_kit.yaml --cfg_assets configs/assets.yaml --batch_size 32 --nodebug
```

## Running the Demo

To run the demo, follow these steps:

1. Download Blender from [Blender Official Website](https://www.blender.org/download/), and place it in the Blender folder within the root directory.
2. Download the pre-trained model from [Pre-Trained Model Link](https://drive.google.com/file/d/1Y7Ht4zmdRbSRLYU41naI2wWLrlW_ZVT0/view?usp=sharing) and put it in the `pre-trained` folder in the root directory.
3. Run the demo by executing `python demo.py --cfg ./configs/config_motiontext_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example demo.txt` with the desired input voice. 
