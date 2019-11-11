using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using CNTK;
using CNTKUtil;

namespace TaxiFarePrediction
{
    /// <summary>
    /// The TaxiTrip class represents a single taxi trip.
    /// </summary>
    public class TaxiTrip
    {
        [LoadColumn(0)] public float VendorId;
        [LoadColumn(5)] public float RateCode;
        [LoadColumn(3)] public float PassengerCount;
        [LoadColumn(4)] public float TripDistance;
        [LoadColumn(9)] public float PaymentType;
        [LoadColumn(10)] public float FareAmount;

        public float[] GetFeatures() => new float[] { VendorId, RateCode, PassengerCount, TripDistance, PaymentType };

        public float GetLabel() => FareAmount;
    }

    /// <summary>
    /// The program class.
    /// </summary>
    class Program
    {
        // file paths to data files
        static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "yellow_tripdata_2018-12.csv");

        /// <summary>
        /// The main application entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        static void Main(string[] args)
        {
            // create the machine learning context
            var context = new MLContext();

            // set up the text loader 
            var textLoader = context.Data.CreateTextLoader(
                new TextLoader.Options() 
                {
                    Separators = new[] { ',' },
                    HasHeader = true,
                    Columns = new[] 
                    {
                        new TextLoader.Column("VendorId", DataKind.Single, 0),
                        new TextLoader.Column("RateCode", DataKind.Single, 5),
                        new TextLoader.Column("PassengerCount", DataKind.Single, 3),
                        new TextLoader.Column("TripDistance", DataKind.Single, 4),
                        new TextLoader.Column("PaymentType", DataKind.Single, 9),
                        new TextLoader.Column("FareAmount", DataKind.Single, 10)
                    }
                }
            );

            // load the data 
            Console.Write("Loading training data....");
            var dataView = textLoader.Load(dataPath);
            Console.WriteLine("done");

            // load training data
            var training = context.Data.CreateEnumerable<TaxiTrip>(dataView, reuseRowObject: false);

            // set up data arrays
            var training_data = training.Select(v => v.GetFeatures()).ToArray();
            var training_labels = training.Select(v => v.GetLabel()).ToArray();

            // build features and labels
            var features = NetUtil.Var(new int[] { 5 }, DataType.Float);
            var labels = NetUtil.Var(new int[] { 1 }, DataType.Float);

            // build a regression model
            var network = features
                .Dense(1)
                .ToNetwork();
            Console.WriteLine("Model architecture:");
            Console.WriteLine(network.ToSummary());

            // set up the loss function and the classification error function
            var lossFunc = NetUtil.MeanSquaredError(network.Output, labels);
            var errorFunc = NetUtil.MeanAbsoluteError(network.Output, labels);

            // set up a trainer
            var learner = network.GetAdamLearner(
                learningRateSchedule: (0.001, 1),
                momentumSchedule: (0.9, 1),
                unitGain: false);

            // set up a trainer and an evaluator
            var trainer = network.GetTrainer(learner, lossFunc, errorFunc);

            // train the model
            Console.WriteLine("Epoch\tTrain\tTrain");
            Console.WriteLine("\tLoss\tError");
            Console.WriteLine("-----------------------");
            
            var maxEpochs = 25; // 50;
            var batchSize = 512; // 32;
            var loss = new double[maxEpochs];
            var trainingError = new double[maxEpochs];
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

                    // train the regression model on the batch
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
                Console.WriteLine($"{epoch}\t{loss[epoch]:F3}\t{trainingError[epoch]:F3}");
            }

            // show final results
            var finalError = trainingError[maxEpochs-1];
            Console.WriteLine();
            Console.WriteLine($"Final MAE: {finalError:0.00}");
        }
    }
}
