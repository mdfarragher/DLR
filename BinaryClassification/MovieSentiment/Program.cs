using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using CNTK;
using CNTKUtil;
using XPlot.Plotly;

namespace MovieSentiment
{
    /// <summary>
    /// The ReviewData class contains one single movie review which may be positive or negative.
    /// </summary>
    public class ReviewData
    {
        [LoadColumn(0)] public string Review { get; set; }
        [LoadColumn(1)] public string Label { get; set; }
    }

    /// <summary>
    /// The ProcessedData class contains one single movie review which has been processed.
    /// </summary>
    public class ProcessedData
    {
        public string Label { get; set; }
        public VBuffer<float> Features { get; set; }

        public float[] GetFeatures() => Features.DenseValues().ToArray();

        public float GetLabel() => Label == "positive" ? 1.0f : 0.0f;
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    public class Program
    {
        // filenames for data set
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "IMDB Dataset.csv");

        /// <summary>
        /// The main program entry point.
        /// </summary>
        /// <param name="args">The command line parameters.</param>
        static void Main(string[] args)
        {
            // set up a machine learning context
            var context = new MLContext();

            // load the dataset in memory
            Console.WriteLine("Loading data...");
            var data = context.Data.LoadFromTextFile<ReviewData>(
                path: dataPath, 
                hasHeader: true, 
                separatorChar: ',',
                allowQuoting: true);

            // use 80% for training and 20% for testing
            var partitions = context.Data.TrainTestSplit(data, testFraction: 0.2);

            // set up a pipeline to featurize the text
            Console.WriteLine("Featurizing text...");
            var pipeline = context.Transforms.Text.ProduceWordBags(
                    outputColumnName: "Features", 
                    inputColumnName: nameof(ReviewData.Review),
                    ngramLength: 1,
                    maximumNgramsCount: 2000);

            // create a model
            var model = pipeline.Fit(partitions.TrainSet);

            // create training and testing datasets 
            var trainingData = model.Transform(partitions.TrainSet);
            var testingData = model.Transform(partitions.TestSet);

            // create training and testing enumerations
            var training = context.Data.CreateEnumerable<ProcessedData>(trainingData, reuseRowObject: false).ToArray();
            var testing = context.Data.CreateEnumerable<ProcessedData>(testingData, reuseRowObject: false).ToArray();

            // set up data arrays
            var training_data = training.Select(v => v.GetFeatures()).ToArray();
            var training_labels = training.Select(v => v.GetLabel()).ToArray();
            var testing_data = testing.Select(v => v.GetFeatures()).ToArray();
            var testing_labels = testing.Select(v => v.GetLabel()).ToArray();

            // calculate width of word vector
            var inputWidth = training_data.First().Length;

            // build features and labels
            var features = NetUtil.Var(new int[] { 2000 }, DataType.Float);
            var labels = NetUtil.Var(new int[] { 1 }, DataType.Float);

            // build the network
            var network = features
                .OneHotOp(10000, true)
                .Embedding(128)
                .TransposeAxes(new CNTK.Axis(1), new CNTK.Axis(0))
                .Convolution1D(32, 7, activation: CNTK.CNTKLib.ReLU)
                .Pooling(PoolingType.Max, new int[] { 5 }, new int[] { 5 })
                .Convolution1D(32, 7, activation: CNTK.CNTKLib.ReLU)
                .Pooling(PoolingType.Max, CNTK.NDShape.Unknown(), new int[] { 5 })
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
            
            var maxEpochs = 10;
            var batchSize = 32;
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
                        y = trainingError.Select(v => 1 - v),
                        name = "training",
                        mode = "lines+markers"
                    },
                    new Graph.Scatter()
                    {
                        x = Enumerable.Range(0, maxEpochs).ToArray(),
                        y = testingError.Select(v => 1 - v),
                        name = "testing",
                        mode = "lines+markers"
                    }
                }
            );
            chart.WithOptions(new Layout.Layout() 
            {
                yaxis = new Graph.Yaxis()
                {
                    rangemode = "tozero"
                }
            });
            chart.WithXTitle("Epoch");
            chart.WithYTitle("Accuracy");
            chart.WithTitle("Movie Review Sentiment");

            // save chart
            File.WriteAllText("chart.html", chart.GetHtml());
        }
    }
}
