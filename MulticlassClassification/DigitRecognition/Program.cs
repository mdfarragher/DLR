using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using CNTK;
using CNTKUtil;
using XPlot.Plotly;

namespace Mnist
{
    /// <summary>
    /// The Digit class represents one mnist digit.
    /// </summary>
    class Digit
    {
        [ColumnName("PixelValues")]
        [VectorType(784)]
        public float[] PixelValues = default;

        [LoadColumn(0)]
        public float Number = default;

        public float[] GetFeatures() => PixelValues;

        public float[] GetLabel() => new float[] {
            Number == 0 ? 1.0f : 0.0f,
            Number == 1 ? 1.0f : 0.0f,
            Number == 2 ? 1.0f : 0.0f,
            Number == 3 ? 1.0f : 0.0f,
            Number == 4 ? 1.0f : 0.0f,
            Number == 5 ? 1.0f : 0.0f,
            Number == 6 ? 1.0f : 0.0f,
            Number == 7 ? 1.0f : 0.0f,
            Number == 8 ? 1.0f : 0.0f,
            Number == 9 ? 1.0f : 0.0f,
        };
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    class Program
    {
        // filenames for data set
        private static string trainDataPath = Path.Combine(Environment.CurrentDirectory, "mnist_train.csv");
        private static string testDataPath = Path.Combine(Environment.CurrentDirectory, "mnist_test.csv");

        /// <summary>
        /// The main program entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        static void Main(string[] args)
        {
            // create a machine learning context
            var context = new MLContext();

            // load data
            Console.WriteLine("Loading data....");
            var columnDef = new TextLoader.Column[]
            {
                new TextLoader.Column(nameof(Digit.PixelValues), DataKind.Single, 1, 784),
                new TextLoader.Column(nameof(Digit.Number), DataKind.Single, 0)
            };
            var trainDataView = context.Data.LoadFromTextFile(
                path: trainDataPath,
                columns : columnDef,
                hasHeader : true,
                separatorChar : ',');
            var testDataView = context.Data.LoadFromTextFile(
                path: testDataPath,
                columns : columnDef,
                hasHeader : true,
                separatorChar : ',');

            // load training and testing data
            var training = context.Data.CreateEnumerable<Digit>(trainDataView, reuseRowObject: false);
            var testing = context.Data.CreateEnumerable<Digit>(testDataView, reuseRowObject: false);

            // set up data arrays
            var training_data = training.Select(v => v.GetFeatures()).ToArray();
            var training_labels = training.Select(v => v.GetLabel()).ToArray();
            var testing_data = testing.Select(v => v.GetFeatures()).ToArray();
            var testing_labels = testing.Select(v => v.GetLabel()).ToArray();

            // build features and labels
            var features = NetUtil.Var(new int[] { 28, 28 }, DataType.Float);
            var labels = NetUtil.Var(new int[] { 10 }, DataType.Float);

            // build the network
            var network = features
                .Dense(512, CNTKLib.ReLU)
                .Dense(10, CNTKLib.Softmax)
                .ToNetwork();
            Console.WriteLine("Model architecture:");
            Console.WriteLine(network.ToSummary());

            // set up the loss function and the classification error function
            var lossFunc = CNTKLib.CrossEntropyWithSoftmax(network.Output, labels);
            var errorFunc = CNTKLib.ClassificationError(network.Output, labels);

            // set up a trainer that uses the RMSProp algorithm
            var learner = network.GetRMSPropLearner(
                learningRateSchedule: 0.99,
                gamma: 0.95,
                inc: 2.0,
                dec: 0.5,
                max: 2.0,
                min: 0.5
            );

            // set up a trainer and an evaluator
            var trainer = network.GetTrainer(learner, lossFunc, errorFunc);
            var evaluator = network.GetEvaluator(errorFunc);

            // train the model
            Console.WriteLine("Epoch\tTrain\tTrain\tTest");
            Console.WriteLine("\tLoss\tError\tError");
            Console.WriteLine("-----------------------------");
            
            var maxEpochs = 50;
            var batchSize = 128;
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
            chart.WithTitle("Digit Training");

            // save chart
            File.WriteAllText("chart.html", chart.GetHtml());
        }
    }

}
