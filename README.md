## [U-CFR: Uncertainty-Guided Cascade Forward Refinement for Interactive Segmentation]()

<p align="center">
  <img src="./assets/img/model_n_icpr.png" alt="drawing"/>
</p>


## Environment
Training and evaluation environment: Python 3.9, PyTorch 1.13.1, CUDA 11.0/11.7. Run the following command to install the required packages.
```bash
conda create -n iseg python=3.9
conda activate iseg
pip3 install -r requirements.txt
```

> [!NOTE]
> **Cython Compilation**: The project uses Cython for fast distance map calculation (`isegm/utils/cython/_get_dist_maps.pyx`). This code is compiled **automatically at runtime** on the first import via `pyximport`—no manual compilation step is required. Ensure that a standard C++ compiler (e.g., GCC, Clang, or MSVC) is installed on your system.

Before training or evaluation, update the dataset paths in [config.yml](/config.yml) to point to your local dataset directories.

## Datasets
Please download the desired evaluation datasets using the links in the table below and configure their paths in `config.yml`.

| Dataset   |                      Description             |           Download Link              |
|-----------|----------------------------------------------|:------------------------------------:|
|SBD        |  8498 images with 20172 instances for (train)<br>2857 images with 6671 instances for (test) |[official site][SBD]|
|Grab Cut   |  50 images with one object each (test)       |  [GrabCut.zip (11 MB)][GrabCut]      |
|Berkeley   |  96 images with 100 instances (test)         |  [Berkeley.zip (7 MB)][Berkeley]     |
|DAVIS      |  345 images with one object each (test)      |  [DAVIS.zip (43 MB)][DAVIS]          |
|Pascal VOC |  1449 images with 3417 instances (test)      |  [official site][PascalVOC]          |
|COCO_MVal  |  800 images with 800 instances (test)        |  [COCO_MVal.zip (127 MB)][COCO_MVal] |

[MSCOCO]: https://cocodataset.org/#download
[LVIS]: https://www.lvisdataset.org/dataset
[SBD]: http://home.bharathh.info/pubs/codes/SBD/download.html
[GrabCut]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip
[Berkeley]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/Berkeley.zip
[DAVIS]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/DAVIS.zip
[PascalVOC]: http://host.robots.ox.ac.uk/pascal/VOC/
[COCOLVIS_annotation]: https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/cocolvis_annotation.tar.gz
[COCO_MVal]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/COCO_MVal.zip

## Demo
<p align="center">
  <img src="./assets/img/demo1.gif" alt="drawing", width="500"/>
</p>

You can launch the interactive demo using the Tkinter interface.

```bash
python demo.py --checkpoint=weights/sbd_vit_base_ufcr.pth --cpu
```

## Evaluation

Before evaluation, please download the datasets and models, and then configure the path in `config.yml`.

Download our trained model:
- [sbd_vit_base_ufcr.pth](https://github.com/elidandar/UCFR-Interactive-Segmentation/releases/download/v0.1.0/sbd_vit_base_ufcr.pth) (377 MB)

Save the weight file to the `weights/` directory. Alternatively, you can download it via the script:
```bash
python weights/download_models.py
```



Use the following code to evaluate the model.

```
python scripts/evaluate_model.py NoBRS \
    --gpu=0 \
    --checkpoint=sbd_vit_base_ufcr.pth \
    --datasets=GrabCut,Berkeley,DAVIS,BraTS,OAIZIB,ssTEM,COCO_MVal,PascalVOC,SBD \
    --cf-n=1 \
    --cf-click 

# cf-n: CFR steps
# cf-click: whether to do ucfr clicks

or submit the evaluation job to the cluster with the script ./iseg_evaluate.sh
```

## Training

Before training, please download the [MAE](https://github.com/facebookresearch/mae) pretrained weights (click to download: [ViT-Base](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth), [ViT-Large](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth), [ViT-Huge](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth)) and configure the dowloaded path in `config.yml`

Please also download the pretrained SimpleClick models from [here](https://github.com/uncbiag/SimpleClick).

Use the following code to train a base model on SBD: 
```
python train.py models/plainvit_base448_sbd.py \
    --batch-size=140 \
    --ngpus=4

or 
Submit the training job to the cluster with the script ./iseg_train.sh
```

## Citation

```

```

## Acknowledgement
Our project is developed based on [SimpleClick](https://github.com/uncbiag/SimpleClick) and [ICL-CFR](https://github.com/TitorX/CFR-ICL-Interactive-Segmentation)
