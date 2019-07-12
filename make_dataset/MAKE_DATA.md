## Make Dataset
---

This task is aimed at cleaning and splitting the downloaded dataset in preparation for model training (in the next task). 

Requirements
- pandas
- numpy
- scikit-learn
- click modules

---

This task is runnable from the command line:

`python make_dataset.py --in-csv --out-dir`

*--in-csv* : is the parameter for the csv file in  ./data_root
*--out-dir* : is the parameter for the directory to which the data will be saved

During the basic analysis of the dataset a couple of columns were discovered to be less impactful hence were dropped. Since column `points` is the column containing information about how valuable the wine is, hence column `rating` was created to rank them:

- 80-85 : 1
- 86-90 : 2
- 91-95 : 3
- 96-100 : 4

Also `pd.get_dummies` was used to encode categorical features so that they can righlty formatted for modeling by the machine learning algorithm in the next phase. This could have been done after the splitting but was done before the splitting because most of the features have a lot categories, in which some appear once. Since the data is splitted, some of those categories that appear once usually fall into just one of the splits either which will result in feature difference hence the model won't be able to predict on tet features.


The output of this task are four csv files namely: `X_train.csv` `y_train.csv` `X_test.csv` `y_test.csv` saved in `/data_root/split_data`
The test data set is 30% of the whole dataset.

---

- X_train: Independent features for the training set
- y_train: Dependent feature for the training set
- X_test: Independent feature for the test set
- y_test: Dependent variable feature for the test set
