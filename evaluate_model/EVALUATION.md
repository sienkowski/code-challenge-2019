## Evaluation
---

This task is aimed at  evaluating the trained model.

Requirements
- Click
- joblib
- matplotlib
- numpy
- pandas
- scikit-learn
- seaborn

---

This task is runnable from the command line:

`python evaluate.py --x-test --y-test --model-path --in-dir  --out-dir`

*--x-test* : is the X_test with the `.csv` extension
*--in-dir* : is the path of the splitted dataset in  `/data_root/train_model`
*--y-test* : is the y_test with the `.csv`
*--out-dir* : is the parameter for the directory to which the image will be saved
*--model-path* : is the path to the model


Here the 30% test data held out is used to evaluate the model. Two metrics are used:
- Confusion Matrix
- Accuracy Score

Confusion Matrix: A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. The confusion matrix usually contain:

- True Positive: Predicted correctly to be positive
- False Positive: Predicted to be positive but actually negative
- True Negative: Predicted correctly to be negative
- False Negative: Wrongly Predicted to be false

For this dataset since the y_train and y_test contain 4 categories the matrix will be  a **4 x 4** matrix. It's usually n X n, where n is the numbe of categories.

- Position 1 x 1: is for the true positive of `rating 1`, hence 1 X 2 to 1 X 4 (row) will be false positives for rating 1. While Position 2 X 1 TO 4 X 1 (column)will be false negatives

- Positon 2 x 2: is for the true positve of `rating 2`, hence its neighbouring rows are false postives and neighbouring columns are false negatives (relative to Position 2 X 1)

- Positon 3 x 3: is for the true positve of `rating 3`, hence its neighbouring rows are false postives and neighbouring columns are false negatives (relative to Position 3 X 3)

- Positon 4 x 4: is for the true positve of `rating 4`, hence its neighbouring rows are false postives and neighbouring columns are false negatives (relative to Position 4 X 4)


Accuracy Score: Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of predictions our model got right. Formally, accuracy has the following definition:
Accuracy = No. of correct predictions/Total No. of predictions

The output of this task is an image showing a confusion matrix with the *accurcy score* as the title. It is saved in `/data_root/evaluate_model`