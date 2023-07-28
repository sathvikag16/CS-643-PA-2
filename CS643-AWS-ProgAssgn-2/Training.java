import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityPrediction {
    
    public static void main(String[] args) {
        // Create Spark Session
        SparkSession spark = SparkSession
            .builder()
            .appName("WineQualityPrediction")
            .getOrCreate();

        // Load Training Data
        Dataset<Row> trainingData = spark.read().format("csv").option("header", "true").load("TrainingDataset.csv");

        // Define Logistic Regression Model
        LogisticRegression logisticRegression = new LogisticRegression()
            .setMaxIter(10)
            .setRegParam(0.3)
            .setElasticNetParam(0.8);

        // Train the Model
        LogisticRegressionModel model = logisticRegression.fit(trainingData);

        // Load Validation Data
        Dataset<Row> validationData = spark.read().format("csv").option("header", "true").load("ValidationDataset.csv");

        // Make Predictions
        Dataset<Row> predictions = model.transform(validationData);

        // Evaluate Model
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("f1");
        double f1Score = evaluator.evaluate(predictions);

        // Output F1 Score
        System.out.println("F1 Score = " + f1Score);

        // Save the Model
        //model.save("path/to/save/model");
    }
}
