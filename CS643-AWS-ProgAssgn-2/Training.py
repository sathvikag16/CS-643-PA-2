#Sharukya Smitesh Marneni 
#CS-643 Programming Assignment - 2 
#Training
import argparse
import os
import quinn
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession

if __name__ == "__main__":
    def clean_up_quotes(dataframe):
        return dataframe.replace('"', '')

    inputCols=["fixed acidity", \
                   "volatile acidity", \
                   "citric acid", \
                   "residual sugar", \
                   "chlorides", \
                   "free sulfur dioxide", \
                   "total sulfur dioxide", \
                   "density", \
                   "pH", \
                   "sulphates", \
                   "alcohol"]
    assembleroutputCol="num_features"
    scaleroutputCol="features"

    spark = SparkSession.builder\
        .appName("cloudwinequality") \
        .getOrCreate()

    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, \
        help="Path to train dataset")
    parser.add_argument('-o', default= os.getcwd() + "/model", \
        help="Path to model output")
    
    args = parser.parse_args()

    ifile = args.i
    ofile = args.o

    trainingDF = spark.read.load(ifile, format="csv", sep=";", inferSchema="true", header="true")
    
    trainingDF= quinn.with_columns_renamed(clean_up_quotes)(trainingDF)
    trainingDF = trainingDF.withColumnRenamed('quality', 'label')

    rf = RandomForestClassifier()
    assembler = VectorAssembler( \
        inputCols=inputCols, \
        outputCol=assembleroutputCol)
    scaler = StandardScaler(inputCol=assembleroutputCol, outputCol=scaleroutputCol, withStd=True)
    
    pipeline = Pipeline(stages=[assembler, scaler, rf])
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 100, 500]) \
        .build()
        
    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(metricName='f1'),
                          numFolds=3)
    
    model = crossval.fit(trainingDF)

    trainingDF.show()
    model = model.bestModel
    print(ofile)
    print(ifile)
    model.write().overwrite().save(ofile)

    evaluator = MulticlassClassificationEvaluator(metricName="f1")

    f1score = evaluator.evaluate(model.transform(trainingDF))
    print("F1 Score for the Model: ", f1score )


    spark.stop()
