from complete_pipeline import CompleteTransformer
import pandas as pd

df=pd.read_csv("D:\\ML project\\data\\archive\\housing.csv")
param=[
    {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
    {'n_estimators':[3,10],'max_features':[2,3,4],'bootstrap':[False]},
]
cmp_transformer = CompleteTransformer(df, "median_house_value", model_name="forest",finetune_params=param)
cmp_transformer.fit()
train_pred, test_pred, grid_search_results, model = cmp_transformer.transform(cmp_transformer.test_processed_data)

print(train_pred)
print(test_pred)
print(grid_search_results.cv_results_)
