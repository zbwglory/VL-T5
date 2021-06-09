# Unifying Vision-and-Language Tasks via Text Generation

* Authors: [Jaemin Cho](https://j-min.io), [Jie Lei](https://www.cs.unc.edu/~jielei/), [Hao Tan](https://www.cs.unc.edu/~airsplay/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
* [Paper](https://arxiv.org/abs/2102.02779) (To appear in [ICML 2021](https://icml.cc/Conferences/2021))


![teaser image](./assets/teaser_square.png)

## Setup
```
# Create python environment (optional)
conda create -n vlt5 python=3.7

# Install python dependencies
pip install -r requirements.txt

# Download T5/BART backbone checkpoint
python download_backbones.py

# For MSCOCO captioning evaluation
python -c "import language_evaluation; language_evaluation.download('coco')"
```

## Code structure
```
# Store images, features, and annotations
./datasets
    COCO/
        images/
        featuers/
    VG/
        images/
        features/
    GQA/
        images/
        features/
    nlvr/
        images/
        features/
    RefCOCO/

    ...

# Run feature extraction
./feature_extraction

# Train VL-T5
./VL-T5/
    src/
        modeling_t5.py modeling_bart.py                       <= VL-T5/VL-BART model classes
        pretrain.py, pretrain_data.py, pretrain_model.py      <= pretraining
        vqa.py, vqa_data.py vqa_model.py ...                  <= fine-tuning on downstream tasks (ex. VQA, GQA, NLVR2)
        multitask.py, multitask_data.py multiask_model.py     <= multitask learning on 7 downstream tasks
        param.py                                              <= (argparse) configuration
        tokenization.py                                       <= custom tokenizer
        utils.py, dist_utils.py                               <= utility functions
    snap/                                                     <= store weight checkpoints
    scripts/                                                  <= bash scripts for pretraining and finetuning
```

## API
```python
import sys
sys.path.append('./VL-T5/src')

# Parse configuration
from param import parse_args
args = parse_args(
    backbone='t5-base' # Backbone architecture
    load='./snap/pretrain/VL-T5/Epoch30' # Pretrained checkpoint
    parse=False, # False for interactive env (ex. jupyter)
)
# Assign GPU
args.gpu = 0

# Load data loaders
from vqa_data import get_loader
train_loader = get_loader(
    args,
    split=args.train,
    ...
)
val_loader = get_loader(
    args,
    split=args.valid,
    ...
)
test_loader = get_loader(
    args,
    split=args.test,
    ...
)

# Import trainer
from vqa import Trainer
trainer = Trainer(
    args,
    train_loader=train_loader
    val_loader=val_loader
    test_loader=test_loader,
)

# model is attached to trainer
model = trainer.model

# Each task-specific model class is inherited from VLT5/VLBart classes, which are inherited from Huggingface transformers T5/BART classes
print(model)
>>> VLT5VQA(
    (shared): Embedding(...)
    (encoder): JointEncoder(...)
    ...
)

# Training
train_batch = next(iter(train_loader))
model.train_step(train_batch)
>>> {'loss': ... }

# Inference
test_batch = next(iter(test_loader))
model.test_step(test_batch)
>>> {'pred_ans': ... }
```

To add a new task, you can start with writing 3 files by editing from existing ones.
```
NEW_TASK_model.py # Define a VLT5NewTask/VLBartNewTask model which inherits VLT5/VLBart class
NEW_TASK_data.py # Define Dataset/DataLoader/Evaluator
NEW_TASK.py # Define a trainer which inherits TrainerBase (trainer_base.py)
```



## Pretrained Models
- Download `snap/` from [Google Drive](https://drive.google.com/drive/folders/1_SBj4sZ0gUqfBon1gFBiNRAmfHv5w_ph?usp=sharing)
* `VL-T5/snap/pretrain/VL-T5/Epoch30.pth`: VL-T5 pretrained for 30 epochs on COCO+VG
* `VL-T5/snap/pretrain/VL-BART/Epoch30.pth`: VL-BART pretrained for 30 epochs on COCO+VG

## Dataset Preparation / Feature extraction
- Download `datasets/` from [Google Drive](https://drive.google.com/drive/folders/1MBBhlkP83VMKS2Qe0SmFfzkHhMpIG5wf?usp=sharing)
  - Multi30K only
    - `git clone --recursive https://github.com/multi30k/dataset ./datasets/multi30k-dataset`
    - unzip `train.en.gz`, `val.en.gz`, `test_2017_flickr.en.gz`, `test_2018_flickr.en.gz` in `./datasets/multi30k-dataset/data/task1/raw/`
    - unzip `train.de.gz`, `val.de.gz`, `test_2017_flickr.de.gz`, `test_2018_flickr.de.gz` in `./datasets/multi30k-dataset/data/task1/raw/`
- For manual feature extraction, please checkout [./feature_extraction](./feature_extraction)

## Pretraining on COCO+VG
```
# with 4 gpus
bash scripts/pretrain_VLT5.sh 4
bash scripts/pretrain_VLBart.sh 4
```

## Downstream tasks
- To be updated

### [VQA](https://visualqa.org/)

### [GQA](https://cs.stanford.edu/people/dorarad/gqa/)

### [NLVR2](http://lil.nlp.cornell.edu/nlvr/)

### [RefCOCOg](https://github.com/mjhucla/Google_Refexp_toolbox)

### [VCR](https://visualcommonsense.com/)

### [COCO Caption](https://cocodataset.org/)

### [Multi30K](https://github.com/multi30k/dataset)


# Reference
Please cite our paper if you use our models in your works:
```bibtex
@inproceedings{cho2020vlt5,
  title     = {Unifying Vision-and-Language Tasks via Text Generation},
  author    = {Jaemin Cho and Jie Lei and Hao Tan and Mohit Bansal},
  booktitle = {ICML},
  year      = {2021}
}
```