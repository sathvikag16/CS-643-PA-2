import random
import sys 
import numpy as np
import pandas as pd
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, desc
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("wine_quality_train").getOrCreate()
training_df = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('./TrainingDataset.csv')
print("Data is loaded")
training_df.printSchema()
training_df.show()
training_df = quinn.with_columns_renamed(remove_quotations)(training_df)
training_df.update(training_df.fillna(training_df.mean()))
catTrainCols = training_df.select_dtypes(include='O')
training_df = pd.get_dummies(training_df, drop_first=True)
training_df['label'] = [1 if x >= 7 else 0 for x in training_df.quality]
validating_df = quinn.with_columns_renamed(remove_quotations)(validating_df)
validating_df.update(validating_df.fillna(validating_df.mean()))
catValCols = validating_df.select_dtypes(include='O')
validating_df = pd.get_dummies(validating_df, drop_first=True)
validating_df['label'] = [1 if x >= 7 else 0 for x in validating_df.quality]
print("Data has been formatted.")
print(training_df.toPandas().head())
assemble = VectorAssembler(
 inputCols=["fixed acidity",
 "volatile acidity",
 "citric acid",
 "residual sugar",
 "chlorides",
 "free sulfur dioxide",
 "total sulfur dioxide",
 "density",
 "pH",
 "sulphates",
 "alcohol"],
 outputCol="inputFeatures")
scaler = Normalizer(inputCol="inputFeatures", outputCol="features")
lr = LogisticRegression()
rf = RandomForestClassifier()
pline1 = Pipeline(stages=[assemble, scaler, lr])
pline2 = Pipeline(stages=[assemble, scaler, rf])
paramgrid = ParamGridBuilder().build()
evaluator = MulticlassClassificationEvaluator(metricName="f1")
crossval = CrossValidator(estimator=pipeline1,estimatorParamMaps=paramgrid,
 evaluator=evaluator, 
 numFolds=3
 )
cvModel1 = crossval.fit(training_df) 
print("F1 Score LogisticRegression Model: ", 
evaluator.evaluate(cvModel1.transform(validating_df)))
crossval = CrossValidator(estimator=pipeline2, 
 estimatorParamMaps=paramgrid,
 evaluator=evaluator, 
 numFolds=3
 )
cvModel2 = crossval.fit(train_df) 
print("F1 Score", evaluator.evaluate(cvModel2.transform(validating_df)))
