using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using CNTK;
using CNTKUtil;
using XPlot.Plotly;

namespace HeartDisease
{
    /// <summary>
    /// The HeartData record holds one single heart data record.
    /// </summary>
    public class HeartData 
    {
        [LoadColumn(0)] public float Age { get; set; }
        [LoadColumn(1)] public float Sex { get; set; }
        [LoadColumn(2)] public float Cp { get; set; }
        [LoadColumn(3)] public float TrestBps { get; set; }
        [LoadColumn(4)] public float Chol { get; set; }
        [LoadColumn(5)] public float Fbs { get; set; }
        [LoadColumn(6)] public float RestEcg { get; set; }
        [LoadColumn(7)] public float Thalac { get; set; }
        [LoadColumn(8)] public float Exang { get; set; }
        [LoadColumn(9)] public float OldPeak { get; set; }
        [LoadColumn(10)] public float Slope { get; set; }
        [LoadColumn(11)] public float Ca { get; set; }
        [LoadColumn(12)] public float Thal { get; set; }
        [LoadColumn(13)] public int Label { get; set; }

        public float[] GetFeatures() => new float[] { Age, Chol, Fbs, Thalac, Exang, OldPeak, Slope };

        public float GetLabel() => (Label == 0 ? 0f : 1f);
    }

    /// <summary>
    /// The application class.
    /// </summary>
    public class Program
    {
        // filenames for training and test data
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "processed.cleveland.data.csv");

        /// <summary>
        /// The main applicaton entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        public static void Main(string[] args)
        {
            // set up a machine learning context
            var context = new MLContext();

            // load training and test data
            Console.WriteLine("Loading data...");
            var data = context.Data.LoadFromTextFile<HeartData>(dataPath, hasHeader: false, separatorChar: ',');

            // split the data into a training and test partition
            var partitions = context.Data.TrainTestSplit(data, testFraction: 0.4);

            // load training and testing data
            var training = context.Data.CreateEnumerable<HeartData>(partitions.TrainSet, reuseRowObject: false);
            var testing = context.Data.CreateEnumerable<HeartData>(partitions.TestSet, reuseRowObject: false);

            // set up data arrays
            var training_data = training.Select(v => v.GetFeatures()).ToArray();
            var training_labels = training.Select(v => v.GetLabel()).ToArray();
            var testing_data = testing.Select(v => v.GetFeatures()).ToArray();
            var testing_labels = testing.Select(v => v.GetLabel()).ToArray();

            // build features and labels
            var features = NetUtil.Var(new int[] { 7 }, DataType.Float);
            var labels = NetUtil.Var(new int[] { 1 }, DataType.Float);

            // build the network
            var network = features
                .Dense(16, CNTKLib.ReLU)
                .Dense(128, CNTKLib.ReLU)
                .Dense(1, CNTKLib.Sigmoid)
                .ToNetwork();

            Console.WriteLine("Model architecture:");
            Console.WriteLine(network.ToSummary());

            // set up the loss function and the classification error function
            var lossFunc = CNTKLib.BinaryCrossEntropy(network.Output, labels);
            var errorFunc = NetUtil.BinaryClassificationError(network.Output, labels);

            // use the Adam learning algorithm
            var learner = network.GetAdamLearner(
                learningRateSchedule: (0.001, 1),
                momentumSchedule: (0.9, 1),
                unitGain: true);

            // set up a trainer and an evaluator
            var trainer = network.GetTrainer(learner, lossFunc, errorFunc);
            var evaluator = network.GetEvaluator(errorFunc);

            // train the model
            Console.WriteLine("Epoch\tTrain\tTrain\tTest");
            Console.WriteLine("\tLoss\tError\tError");
            Console.WriteLine("-----------------------------");
            
            var maxEpochs = 100;
            var batchSize = 1;
            var loss = new double[maxEpochs];
            var trainingError = new double[maxEpochs];
            var testingError = new double[maxEpochs];
            var batchCount = 0;
            for (int epoch = 0; epoch < maxEpochs; epoch++)
            {
                // train one epoch on batches
                loss[epoch] = 0.0;
                trainingError[epoch] = 0.0;
                batchCount = 0;
                training_data.Index().Shuffle().Batch(batchSize, (indices, begin, end) =>
                {
                    // get the current batch
                    var featureBatch = features.GetBatch(training_data, indices, begin, end);
                    var labelBatch = labels.GetBatch(training_labels, indices, begin, end);

                    // train the network on the batch
                    var result = trainer.TrainBatch(
                        new[] {
                            (features, featureBatch),
                            (labels,  labelBatch)
                        },
                        false
                    );
                    loss[epoch] += result.Loss;
                    trainingError[epoch] += result.Evaluation;
                    batchCount++;
                });

                // show results
                loss[epoch] /= batchCount;
                trainingError[epoch] /= batchCount;
                Console.Write($"{epoch}\t{loss[epoch]:F3}\t{trainingError[epoch]:F3}\t");

                // test one epoch on batches
                testingError[epoch] = 0.0;
                batchCount = 0;
                testing_data.Batch(batchSize, (data, begin, end) =>
                {
                    // get the current batch for testing
                    var featureBatch = features.GetBatch(testing_data, begin, end);
                    var labelBatch = labels.GetBatch(testing_labels, begin, end);

                    // test the network on the batch
                    testingError[epoch] += evaluator.TestBatch(
                        new[] {
                            (features, featureBatch),
                            (labels,  labelBatch)
                        }
                    );
                    batchCount++;
                });
                testingError[epoch] /= batchCount;
                Console.WriteLine($"{testingError[epoch]:F3}");
            }

            // show final results
            var finalError = testingError[maxEpochs-1];
            Console.WriteLine();
            Console.WriteLine($"Final test error: {finalError:0.00}");
            Console.WriteLine($"Final test accuracy: {1 - finalError:0.00}");

            // plot the error graph
            var chart = Chart.Plot(
                new [] 
                {
                    new Graph.Scatter()
                    {
                        x = Enumerable.Range(0, maxEpochs).ToArray(),
                        y = trainingError,
                        name = "training",
                        mode = "lines+markers"
                    },
                    new Graph.Scatter()
                    {
                        x = Enumerable.Range(0, maxEpochs).ToArray(),
                        y = testingError,
                        name = "testing",
                        mode = "lines+markers"
                    }
                }
            );
            chart.WithXTitle("Epoch");
            chart.WithYTitle("Classification error");
            chart.WithTitle("Heart Disease Training");

            // save chart
            File.WriteAllText("chart.html", chart.GetHtml());
        }
    }
}
