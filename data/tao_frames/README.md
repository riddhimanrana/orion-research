---
task_categories:
- object-detection
license: mit
tags:
- computer vision
- amodal-tracking
- object-tracking
- amodal-perception
configs:
- config_name: default
  data_files:
  - split: train
    path: "amodal_annotations/train.json"
  - split: validation
    path: "amodal_annotations/validation.json"
  - split: test
    path: "amodal_annotations/test.json"
extra_gated_prompt: "To download the AVA and HACS videos you have to agree to terms and conditions."
extra_gated_fields:
  You will use the Datasets only for non-commercial research and educational purposes.:
    type: select
    options: 
      - Yes
      - No
  You will NOT distribute the Datasets or any parts thereof.:
    type: select
    options: 
      - Yes
      - No
  Carnegie Mellon University makes no representations or warranties regarding the datasets, including but not limited to warranties of non-infringement or fitness for a particular purpose.:
    type: select
    options: 
      - Yes
      - No
  You accept full responsibility for your use of the datasets and shall defend and indemnify Carnegie Mellon University, including its employees, officers and agents, against any and all claims arising from your use of the datasets, including but not limited to your use of any copyrighted videos or images that you may create from the datasets.:
    type: select
    options:
      - Yes
      - No
  You will treat people appearing in this data with respect and dignity.:
    type: select
    options: 
      - Yes
      - No
  This data comes with no warranty or guarantee of any kind, and you accept full liability.:
    type: select
    options: 
      - Yes
      - No
extra_gated_heading: "TAO-Amodal VIDEO Request"
extra_gated_button_content: "Request Data"
---

# TAO-Amodal Dataset

<!-- Provide a quick summary of the dataset. -->
 Official Source for Downloading the TAO-Amodal and TAO Dataset.
   
   [**📙 Project Page**](https://tao-amodal.github.io/)  | [**💻 Code**](https://github.com/WesleyHsieh0806/TAO-Amodal) | [**📎 Paper Link**](https://arxiv.org/abs/2312.12433) | [**✏️ Citations**](#citations)
   
   <div align="center">
  <a href="https://tao-amodal.github.io/"><img width="95%" alt="TAO-Amodal" src="https://tao-amodal.github.io/static/images/webpage_preview.png"></a>
   </div>

</br>

Contact: [🙋🏻‍♂️Cheng-Yen (Wesley) Hsieh](https://wesleyhsieh0806.github.io/)

## Dataset Description
Our dataset augments the TAO dataset with amodal bounding box annotations for fully invisible, out-of-frame, and occluded objects. 
Note that this implies TAO-Amodal also includes modal segmentation masks (as visualized in the color overlays above). 
Our dataset encompasses 880 categories, aimed at assessing the occlusion reasoning capabilities of current trackers 
through the paradigm of Tracking Any Object with Amodal perception (TAO-Amodal).

You can also find the annotations of TAO dataset in `annotations` folder.

### Dataset Download
1. Download with git:
```bash
git lfs install
git clone git@hf.co:datasets/chengyenhsieh/TAO-Amodal
```

- Download with [`python`](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-from-the-hub):

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="chengyenhsieh/TAO-Amodal")
```

2. Unzip all videos:

Modify `dataset_root` in [unzip_video.py](./unzip_video.py) and run:

```bash
python unzip_video.py
```



## 📚 Dataset Structure

The dataset should be structured like this:
```bash
   TAO-Amodal
    ├── frames
    │    └── train
    │       ├── ArgoVerse
    │       ├── BDD
    │       ├── Charades
    │       ├── HACS
    │       ├── LaSOT
    │       └── YFCC100M
    ├── amodal_annotations
    │    ├── train/validation/test.json
    │    ├── train_lvis_v1.json
    │    └── validation_lvis_v1.json
    ├── annotations (TAO annotations)
    │    ├── train/validation.json
    │    ├── train/validation_with_freeform.json
    │    └── README.md
    ├── example_output
    │    └── prediction.json
    ├── BURST_annotations
    │    ├── train
    │         └── train_visibility.json
    │    ...

```

## 📚 File Descriptions

| File Name                  | Description                                                                                                                                                                                                                             |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| train/validation/test.json | Formal annotation files. We use these annotations for visualization. Categories include those in [lvis](https://www.lvisdataset.org/) v0.5 and freeform categories.                                                                     |
| train_lvis_v1.json         | We use this file to train our [amodal-expander](https://tao-amodal.github.io/index.html#Amodal-Expander), treating each image frame as an independent sequence. Categories are aligned with those in lvis v1.0.                         |
| validation_lvis_v1.json    | We use this file to evaluate our [amodal-expander](https://tao-amodal.github.io/index.html#Amodal-Expander). Categories are aligned with those in lvis v1.0.                                                                            |
| prediction.json            | Example output json from amodal-expander. Tracker predictions should be structured like this file to be evaluated with our [evaluation toolkit](https://github.com/WesleyHsieh0806/TAO-Amodal?tab=readme-ov-file#bar_chart-evaluation). |
| BURST_annotations/XXX.json | Modal mask annotations from [BURST dataset](https://github.com/Ali2500/BURST-benchmark) with our heuristic visibility attributes. We provide these files for the convenience of visualization                                           |

### Annotation and Prediction Format

Our annotations are structured similarly as [TAO](https://github.com/TAO-Dataset/tao/blob/master/tao/toolkit/tao/tao.py#L4) with some modifications.
Annotations:
```bash

Annotation file format:
{
    "info" : info,
    "images" : [image],
    "videos": [video],
    "tracks": [track],
    "annotations" : [annotation],
    "categories": [category],
    "licenses" : [license],
}
annotation: {
    "id": int,
    "image_id": int,
    "track_id": int,
    "bbox": [x,y,width,height],
    "area": float,

    # Redundant field for compatibility with COCO scripts
    "category_id": int,
    "video_id": int,

    # Other important attributes for evaluation on TAO-Amodal
    "amodal_bbox": [x,y,width,height],
    "amodal_is_uncertain": bool,
    "visibility": float, (0.~1.0)
}
image, info, video, track, category, licenses, : Same as TAO
```

Predictions should be structured as:

```bash
[{
    "image_id" : int,
    "category_id" : int,
    "bbox" : [x,y,width,height],
    "score" : float,
    "track_id": int,
    "video_id": int
}]
```
Refer to the instructions of [TAO dataset](https://github.com/TAO-Dataset/tao/blob/master/docs/evaluation.md) for further details


## 📺 Example Sequences
Check [here](https://tao-amodal.github.io/#TAO-Amodal) for more examples and [here](https://github.com/WesleyHsieh0806/TAO-Amodal?tab=readme-ov-file#artist-visualization) for visualization code.
[<img src="https://tao-amodal.github.io/static/images/car_and_bus.png" width="50%">](https://tao-amodal.github.io/dataset.html "tao-amodal")



## Citation 

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->
```
@article{hsieh2023tracking,
          title={Tracking any object amodally},
          author={Hsieh, Cheng-Yen and Khurana, Tarasha and Dave, Achal and Ramanan, Deva},
          journal={arXiv preprint arXiv:2312.12433},
          year={2023}
        }
```

<details>
  <summary>Please also cite <a href="https://taodataset.org/">TAO</a> and <a href="https://github.com/Ali2500/BURST-benchmark">BURST</a> dataset if you use our dataset</summary>

  ```
@inproceedings{dave2020tao,
    title={Tao: A large-scale benchmark for tracking any object},
    author={Dave, Achal and Khurana, Tarasha and Tokmakov, Pavel and Schmid, Cordelia and Ramanan, Deva},
    booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part V 16},
    pages={436--454},
    year={2020},
    organization={Springer}
  }

@inproceedings{athar2023burst,
  title={Burst: A benchmark for unifying object recognition, segmentation and tracking in video},
  author={Athar, Ali and Luiten, Jonathon and Voigtlaender, Paul and Khurana, Tarasha and Dave, Achal and Leibe, Bastian and Ramanan, Deva},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1674--1683},
  year={2023}
}
  ```

</details>

