## Background
U.S. Federal agencies are required to report descriptive free text information
about each Federal award to USAspending using the Award Description element.
Award descriptions are requried to include a description of the
purpose of the award.  In practice, the values provided in this element often do
not clearly include this information
([M-22-02](https://www.whitehouse.gov/wp-content/uploads/2021/10/M-22-02.pdf)).

This project uses NLP and machine learning to develop a model that can be used
to determine whether award descriptions meet this requirement. 

## Procedure
1. Download all prime award loans worth over $1M from USAspending advanced search.    
    - about 100,000 awards
    - about 300 unique award descriptions
2. Manually apply the dependent variable to unique award description: 
    - a boolean flag that indicates whether the description includes the purpose
    of the award.
    - 54% of award descriptions in the test set included the purpose of the award.
    - store tagged descriptions as `./tagged_award_descriptions.csv`
3. Split data into train and test sets.
4. Use sklearn `TfidfTransformer` and `CountVectorizer` to extract features for NLP analysis.
5. Use sklearn `Pipeline` and `GridSearchCV` to tune KNN and SVC models using
training data, 5 folds cross validation, and f1 score.
6. Apply the best KNN and SVC models to test set. 

## Results
SVC provides a better test set accuracy, precision, and recal.

The tables below illustrate the test set performance of the best SVC and KNN models.


### Model Performance Comparison

| Metric         | SVC   | KNN   |
|----------------|-------|-------|
| **Accuracy**   | 0.86  | 0.73  |
| **Precision**  | 0.88  | 0.74  |
| **Recall**     | 0.86  | 0.73  |

### Detailed Class-wise Performance

#### SVC Model
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.95      | 0.73   | 0.83     | 26      |
| 1     | 0.82      | 0.97   | 0.89     | 33      |
| **Macro Avg**  | 0.89      | 0.85   | 0.86     | 59      |
| **Weighted Avg** | 0.88    | 0.86   | 0.86     | 59      |

#### KNN Model
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.78      | 0.54   | 0.64     | 26      |
| 1     | 0.71      | 0.88   | 0.78     | 33      |
| **Macro Avg**  | 0.74      | 0.71   | 0.72     | 59      |
| **Weighted Avg** | 0.74    | 0.73   | 0.72     | 59      |


## Discussion
- Later experimentation has shown that modern LLMs are capable of correctly
making this classification.  
- This model performed well in identifying awards that did not include the
purpose of the award.  
- Other techniques not evaluated might outperform SVC (Random Forest, XGBoost).
- Other variables besides the processed award descriptions could improve the
model, such as the Federal agency that made the award.
- I manually tagged the awards myself. This certainly biased the results.