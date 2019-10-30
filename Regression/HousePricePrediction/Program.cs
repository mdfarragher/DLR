using System;
using System.IO;
using System.Linq;
using CNTK;
using CNTKUtil;
using XPlot.Plotly;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HousePricePrediction
{
   /// <summary>
    /// The HouseBlockData class holds one single housing block data record.
    /// </summary>
    public class HouseBlockData
    {
        [LoadColumn(0)] public float Longitude { get; set; }
        [LoadColumn(1)] public float Latitude { get; set; }
        [LoadColumn(2)] public float HousingMedianAge { get; set; }
        [LoadColumn(3)] public float TotalRooms { get; set; }
        [LoadColumn(4)] public float TotalBedrooms { get; set; }
        [LoadColumn(5)] public float Population { get; set; }
        [LoadColumn(6)] public float Households { get; set; }
        [LoadColumn(7)] public float MedianIncome { get; set; }
        [LoadColumn(8)] public float MedianHouseValue { get; set; }

        public float[] GetFeatures() => new float[] { Longitude, Latitude, HousingMedianAge, TotalRooms, TotalBedrooms, Population, Households, MedianIncome }; 

        public float GetLabel() => MedianHouseValue / 1000.0f;
    }

    /// <summary>
    /// HouseTrainingEngine is a custom training engine for this assignment.
    /// </summary>
    class HouseTrainingEngine : TrainingEngine
    {
        /// <summary>
        /// Set up the feature variable.
        /// </summary>
        /// <returns>The feature variable to use.</returns>
        protected override Variable CreateFeatureVariable()
        {
            return NetUtil.Var(new int[] { 8 }, DataType.Float);
        }

        /// <summary>
        /// Set up the label variable.
        /// </summary>
        /// <returns>The label variable to use.</returns>
        protected override Variable CreateLabelVariable()
        {
            return NetUtil.Var(new int[] { 1 }, DataType.Float);
        }

        /// <summary>
        /// Set up the model.
        /// </summary>
        /// <param name="features">The input feature to use.</param>
        /// <returns>The completed model.</returns>
        protected override Function CreateModel(Variable features)
        {
            return features
                .Dense(8, CNTKLib.ReLU)
                .Dense(8, CNTKLib.ReLU)
                .Dense(1)
                .ToNetwork();
        }
    }

    class Program
    {
        /// <summary>
        /// The main entry point of the application.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        [STAThread]
        public static void Main(string[] args)
        {
            // create the machine learning context
            var context = new MLContext();

            // load the dataset
            Console.WriteLine("Loading data...");
            var data = context.Data.LoadFromTextFile<HouseBlockData>(
                path: "california_housing.csv", 
                hasHeader:true, 
                separatorChar: ',');

            // split into training and testing partitions
            var partitions = context.Data.TrainTestSplit(data, 0.2);

            // load training and testing data
            var training = context.Data.CreateEnumerable<HouseBlockData>(partitions.TrainSet, reuseRowObject: false);
            var testing = context.Data.CreateEnumerable<HouseBlockData>(partitions.TestSet, reuseRowObject: false);

            // set up a new training engine
            Console.WriteLine("Setting up training engine...");
            var engine = new HouseTrainingEngine()
            {
                LossFunction = TrainingEngine.LossFunctionType.MSE,
                MetricType = TrainingEngine.MetricTypeEnum.Loss,
                NumberOfEpochs = 50,
                BatchSize = 16,
                LearningRate = 0.001
            };

            // load the data into the engine
            engine.SetData(
                training.Select(v => v.GetFeatures()).ToArray(), 
                training.Select(v => v.GetLabel()).ToArray(), 
                testing.Select(v => v.GetFeatures()).ToArray(), 
                testing.Select(v => v.GetLabel()).ToArray());

            // start the training
            Console.WriteLine("Start training...");
            engine.Train();

            // plot training and testing curves
            var chart = Chart.Plot(
                new [] 
                {
                    new Graph.Scatter()
                    {
                        x = engine.TrainingCurve.Select(value => value.X).ToArray(),
                        y = engine.TrainingCurve.Select(value => Math.Sqrt(value.Y)).ToArray(),
                        name = "training",
                        mode = "lines+markers"
                    },
                    new Graph.Scatter()
                    {
                        x = engine.TestingCurve.Select(value => value.X).ToArray(),
                        y = engine.TestingCurve.Select(value => Math.Sqrt(value.Y)).ToArray(),
                        name = "testing",
                        mode = "lines+markers"
                    }
                }
            );
            chart.WithXTitle("Epoch");
            chart.WithYTitle("Loss (RMSE)");
            chart.WithTitle("California Housing Training");

            // save chart
            File.WriteAllText("chart.html", chart.GetHtml());
        }
    }
}
