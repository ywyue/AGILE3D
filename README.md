<p align="center">
<h1 align="center">AGILE3D: Attention Guided Interactive Multi-object 3D Segmentation</h1>
<p align="center">
<a href="https://n.ethz.ch/~yuayue/"><strong>Yuanwen Yue</strong></a>
,
<a href="https://www.vision.rwth-aachen.de/person/218/"><strong>Sabarinath Mahadevan</strong></a>
,
<a href="https://jonasschult.github.io/"><strong>Jonas Schult</strong></a>
,
<a href="https://francisengelmann.github.io/"><strong>Francis Engelmann</strong></a>
<br>
<a href="https://www.vision.rwth-aachen.de/person/1/"><strong>Bastian Leibe</strong></a>
, 
<a href="https://igp.ethz.ch/personen/person-detail.html?persid=143986"><strong>Konrad Schindler</strong></a>
,
<a href="https://theodorakontogianni.github.io/"><strong>Theodora Kontogianni</strong></a>
</p>
<h2 align="center">ICLR 2024</h2>
<h3 align="center"><a href="https://arxiv.org/abs/2306.00977">Paper</a> | <a href="https://ywyue.github.io/AGILE3D/">Project Webpage</a></h3>
</p>
<p align="center">
<img src="./imgs/teaser.gif" width="500"/>
</p>
<p align="center">
<strong>AGILE3D</strong> supports interactive multi-object 3D segmentation, where a user collaborates with a deep learning model to segment multiple 3D objects simultaneously, by providing interactive clicks.
</p>

## News :loudspeaker:

- [2024/01/19] Our interactive segmentation tool was released. Try your own scans! :smiley:
- [2024/01/16] AGILE3D was accepted to ICLR 2024 :tada:


## TODO :memo:
- :white_check_mark: Release pretrained model and our interactive segmentation tool
- :white_large_square: Release training and evaluation code (Expected before 31 January)
- :white_large_square: Release benchmark setup (Expected before 31 January)

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation-hammer">Installation</a>
    </li>
    <li>
      <a href="#interactive-tool-video_game">Interactive Tool</a>
    </li>
    <li>
      <a href="#training-rocket">Training</a>
    </li>
    <li>
      <a href="#evaluation-dart">Evaluation</a>
    </li>
    <li>
      <a href="#citation-mortar_board">Citation</a>
    </li>
    <li>
      <a href="#acknowledgment-pray">Acknowledgment</a>
    </li>
  </ol>
</details>

## Installation :hammer:

Please follow the [installation.md](https://github.com/ywyue/AGILE3D/tree/main/installation.md) to set up the environments.

## Interactive Tool :video_game:

Please follow this [this instruction](https://github.com/ywyue/AGILE3D/tree/main/demo.md) to play with the interactive tool yourself.

<p align="center">
<img src="./imgs/demo.gif" width="75%" />
</p>

We present an **interactive** tool that allows users to segment/annotate **multiple 3D objects** together, in an **open-world** setting. Although our work focuses on multi-object cases, this tool can also support interactive single-object segmentation seamlessly. Although the model was only trained on ScanNet training set, it can also segment unseen datasets like S3DIS, ARKitScenes, and even outdoor scans like KITTI-360. Please check the [project page](https://ywyue.github.io/AGILE3D/) for more demos. Also try your own scans :smiley:

## Training :rocket:

Coming soon.

## Evaluation :dart:

Coming soon.

## Citation :mortar_board:

If you find our code or paper useful, please cite:

```
@inproceedings{yue2023agile3d,
  title     = {{AGILE3D: Attention Guided Interactive Multi-object 3D Segmentation}},
  author    = {Yue, Yuanwen and Mahadevan, Sabarinath and Schult, Jonas and Engelmann, Francis and Leibe, Bastian and Schindler, Konrad and Kontogianni, Theodora},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}
```

## Acknowledgment :pray:

**We sincerely thank all volunteers who participated in our user study!** Francis Engelmann and Theodora Kontogianni are postdoctoral research fellows at the ETH AI Center. This project is partially funded by the ETH Career Seed Award - Towards Open-World 3D Scene Understanding,
NeuroSys-D (03ZU1106DA) and BMBF projects 6GEM (16KISK036K).

Parts of our code are built on top of [Mask3D](https://github.com/JonasSchult/Mask3D) and [InterObject3D](https://github.com/theodorakontogianni/InterObject3D). We also thank Anne Marx for the help in the initial version of the GUI.
