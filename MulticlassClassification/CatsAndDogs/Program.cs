using System.Runtime.CompilerServices;
using System;
using System.IO;
using System.Linq;
using CNTK;
using CNTKUtil;
using XPlot.Plotly;

namespace CatsAndDogs
{
    /// <summary>
    /// The main program class.
    /// </summary>
    class Program
    {
        // filenames for data set
        private static string trainMapPath = Path.Combine(Environment.CurrentDirectory, "train_map.txt");
        private static string testMapPath = Path.Combine(Environment.CurrentDirectory, "test_map.txt");

        // total number of images in the training set
        private const int trainingSetSize = 1600; // 80% of 2000 images
        private const int testingSetSize = 400;   // 20% of 2000 images

        /// <summary>
        /// Create the mapping files for features and labels
        /// </summary>
        static void CreateMappingFiles()
        {
            // get both classes of images
            var class0Images = Directory.GetFiles(Path.Combine(Environment.CurrentDirectory, "cat"));
            var class1Images = Directory.GetFiles(Path.Combine(Environment.CurrentDirectory, "dog"));

            // generate train and test mapping files
            var mappingFiles = new string[] { trainMapPath, testMapPath };
            var partitionSizes = new int[] { trainingSetSize, testingSetSize };
            var imageIndex = 0;
            for (int mapIndex = 0; mapIndex < mappingFiles.Length; mapIndex++)
            {
                var filePath = mappingFiles[mapIndex];
                using (var dstFile = new StreamWriter(filePath))
                {
                    for (var i = 0; i < partitionSizes[mapIndex]; i++)
                    {
                        var class0Path = Path.Combine("cat", class0Images[imageIndex]);
                        var class1Path = Path.Combine("dog", class1Images[imageIndex]);
                        dstFile.WriteLine($"{class0Path}\t0");
                        dstFile.WriteLine($"{class1Path}\t1");
                        imageIndex++;
                    }
                }
                Console.WriteLine($"  Created file: {filePath}");
            }
            Console.WriteLine();
        }

        // image details
        private const int imageWidth = 150;
        private const int imageHeight = 150;
        private const int numChannels = 3;

        /// <summary>
        /// The main program entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        static void Main(string[] args)
        {
            // create the mapping files
            Console.WriteLine("Creating mapping files...");
            CreateMappingFiles();

            // download the VGG16 network
            Console.WriteLine("Downloading VGG16...");
            if (!DataUtil.VGG16.IsDownloaded)
            {
                DataUtil.VGG16.Download();
            }

            // get a training and testing image readers
            var trainingReader = DataUtil.GetImageReader(trainMapPath, imageWidth, imageHeight, numChannels, 2, randomizeData: true, augmentData: true);
            var testingReader = DataUtil.GetImageReader(testMapPath, imageWidth, imageHeight, numChannels, 2, randomizeData: false, augmentData: false);

            // build features and labels
            var features = NetUtil.Var(new int[] { imageHeight, imageWidth, numChannels }, DataType.Float);
            var labels = NetUtil.Var(new int[] { 2 }, DataType.Float);

            // build the network
            var network = features
                .MultiplyBy<float>(1.0f / 255.0f)  // divide all pixels by 255
                .VGG16(allowBlock5Finetuning: false)
                .Dense(256, CNTKLib.ReLU)
                .Dropout(0.5)
                .Dense(2, CNTKLib.Softmax)
                .ToNetwork();
            Console.WriteLine("Model architecture:");
            Console.WriteLine(network.ToSummary());

            // set up the loss function and the classification error function
            var lossFunction = CNTKLib.CrossEntropyWithSoftmax(network.Output, labels);
            var errorFunction = CNTKLib.ClassificationError(network.Output, labels);

            // use the Adam learning algorithm
            var learner = network.GetAdamLearner(
                learningRateSchedule: (0.0001, 1),
                momentumSchedule: (0.99, 1));

            // set up a trainer and an evaluator
            var trainer = network.GetTrainer(learner, lossFunction, errorFunction);
            var evaluator = network.GetEvaluator(errorFunction);

            // train the model
            Console.WriteLine("Epoch\tTrain\tTrain\tTest");
            Console.WriteLine("\tLoss\tError\tError");
            Console.WriteLine("-----------------------------");
            
            var maxEpochs = 25;
            var batchSize = 16;
            var loss = new double[maxEpochs];
            var trainingError = new double[maxEpochs];
            var testingError = new double[maxEpochs];
            var batchCount = 0;
            for (int epoch = 0; epoch < maxEpochs; epoch++)
            {
                // train one epoch on batches
                loss[epoch] = 0.0;
                trainingError[epoch] = 0.0;
                var sampleCount = 0;
                while (sampleCount < 2 * trainingSetSize)
                {
                    // get the current batch for training
                    var batch = trainingReader.GetBatch(batchSize);
                    var featuresBatch = batch[trainingReader.StreamInfo("features")];
                    var labelsBatch = batch[trainingReader.StreamInfo("labels")];

                    // train the model on the batch
                    var result = trainer.TrainBatch(
                        new[] {
                            (features, featuresBatch),
                            (labels,  labelsBatch)
                        }
                    );
                    loss[epoch] += result.Loss;
                    trainingError[epoch] += result.Evaluation;
                    sampleCount += (int)featuresBatch.numberOfSamples;
                    batchCount++;                
                }

                // show results
                loss[epoch] /= batchCount;
                trainingError[epoch] /= batchCount;
                Console.Write($"{epoch}\t{loss[epoch]:F3}\t{trainingError[epoch]:F3}\t");

                // test one epoch on batches
                testingError[epoch] = 0.0;
                batchCount = 0;
                sampleCount = 0;
                while (sampleCount < 2 * testingSetSize)
                {
                    // get the current batch for testing
                    var batch = testingReader.GetBatch(batchSize);
                    var featuresBatch = batch[testingReader.StreamInfo("features")];
                    var labelsBatch = batch[testingReader.StreamInfo("labels")];

                    // test the model on the batch
                    testingError[epoch] += evaluator.TestBatch(
                        new[] {
                            (features, featuresBatch),
                            (labels,  labelsBatch)
                        }
                    );
                    sampleCount += (int)featuresBatch.numberOfSamples;
                    batchCount++;
                }

                // show results
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
            chart.WithXTitle("Epoch");
            chart.WithYTitle("Accuracy");
            chart.WithTitle("Cats and Dogs Training");

            // save chart
            File.WriteAllText("chart.html", chart.GetHtml());
        }
    }
}