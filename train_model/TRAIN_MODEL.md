## Train Model
---

This task is aimed at training the model after the data has been cleaned, preprocessed and splitted.

Requirements
- scikit-learn
- pandas
- numpy
- joblib
- click 

---

This task is runnable from the command line:

`python train_model.py --x-train --y-train --in-dir  --out-dir`

*--x-train* : is the X_train with the `.csv` extension
*--in-dir* : is the path of the splitted dataset in  `/data_root/split_data`
*--y-train* : is the y_train with the `.csv`
*--out-dir* : is the parameter for the directory to which the model will be saved `/data_root/model`

Random Forest was the choosen algorithm for this task because:

- Bagged trees are used on X boostrapped training set.
- It can handle binary features, categorical features, and numerical features. 
- The data does not need to be rescaled or transformed.
- They are parallelizable, meaning that we can split the process to multiple machines to run. This results in faster computation time.
- Boosted models are sequential in contrast, and would take longer to compute.


The model was serialized using joblib because:

- joblib is usually significantly faster on large numpy arrays because it has a special handling for the array buffers of the numpy dataset. 

- It can also compress that data on the fly while pickling using zlib or lz4.

- joblib also makes it possible to memory map the data buffer of an uncompressed joblib-pickled numpy array when loading it which makes it possible to share memory between processes.

The output of this task is a `.joblib` serialized file saved in `/data_root/model/`
