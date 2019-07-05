# Perfect Half Million Beauty Product Image Recognition Challenge


This is my solution for [Perfect Half Million Beauty Product Image Recognition Challenge
](https://challenge2019.perfectcorp.com), which obtained the 2nd place (007) with MAP@7 0.407233.


### Requirement
- python 3.6
- pytorch 0.4.0
- PIL
- torchvision [here](https://github.com/Cadene/pretrained-models.pytorch)
- pretrainedmodels
- faiss [here](https://github.com/facebookresearch/faiss)
- download image data from [here](https://challenge2019.perfectcorp.com) if published
and place them in `./data/`  and `./test/`
- place `val.csv` and `test.csv` in `./`

### How to use
- For obtaining the retrieval result, run `bash ./predict`  
	which needs three inputs: "train_images_path", "test_images_path", "predictions_path"
- For training UEL, run `python trainUEL.py`  
  which needs three inputs: "train_path", "test_path", "tlabel"
  - `-train_path `(default: 1) : the file for training data
  - `-test_path `(default: './data/') : the file for val data
  -  `-tlabel `(default: './val.csv') : the label for val data