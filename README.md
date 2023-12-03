# Hotel-reservation-prediction

prediction model of hotel reservation dataset

## Dataset Visualization and Pre-analysis

* ### Distribution analysis
  
  ```from sebadefs import distribution_analysis```
  based on https://github.com/cokelaer/fitter
  
  ## Preprocesing

* ### Scaling
  
  ```sklearn.preprocessing.RobustScaler```<br>
  
  > https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

* ### Categorical Encoding
  
  ```df = pd.get_dummies()```

* ### Under-Sampling
  
  Under-sampling is a technique to balance uneven datasets by keeping all of the data in the minority class and decreasing the size of the majority class.
  NOTE : We should use sampling methods only upon the train dataset otherwise it would be inacurate.

* ### Over-sampling - SVM-SMOTE
  
  SVM-SMOTE, an over-sampling technique, is used to investigate how well it handles the trade-off. SMOTE, its ancestor, is a popular over-sampling technique which balances class distribution by synthetically generating new minority class instances along directions from existing minority class instances towards their nearest neighbours.SVM-SMOTE focuses on generating new minority class instances near borderlines with SVM so as to help establish boundary between classes.<br>
  https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5<br>
  https://arxiv.org/pdf/1106.1813.pdf<br>

## Testing Prediction Models

* ### Random Forest Classifier
* ### XGB
* ### Logistic Regretion
* ### SVC
* ### CatBoost Classifier
* ### Voting Classifier (mix previous best models)

**Note:** *Models were tested first without Under-Sampling and Over-sampling, then best models were tested again with Under-Sampling and Over-sampling preprocessing.*
