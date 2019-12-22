using System;
using System.Collections.Generic;
using CNTK;

namespace CNTKUtil
{
    /// <summary>
    /// The TrainingEngine class encapsulates all code required to train and evaluate a neural network.
    /// </summary>
    public abstract class TrainingEngine
    {
        /// <summary>
        /// The different types of loss function to use during training and evaluation.
        /// </summary>
        public enum LossFunctionType { BinaryCrossEntropy, MSE, CrossEntropyWithSoftmax, CrossEntropyWithSoftmaxWithOneHotEncodedLabel, Custom };

        /// <summary>
        /// The different types of accuracy function to use during training and evaluation.
        /// </summary>
        public enum AccuracyFunctionType { BinaryAccuracy, SameAsLoss };

        /// <summary>
        /// The different types of metric to track during training.
        /// </summary>
        public enum MetricType { Loss, Accuracy }

        /// <summary>
        /// The loss function to use during training and evaluation. The default is BinaryCrossEntropy.
        /// </summary>
        public LossFunctionType lossFunctionType = LossFunctionType.BinaryCrossEntropy;

        /// <summary>
        /// The accuracy function to use during training and evaluation. The default is BinaryAccuracy.
        /// </summary>
        public AccuracyFunctionType accuracyFunctionType = AccuracyFunctionType.BinaryAccuracy;

        /// <summary>
        /// The metric to track during training and evaluation. The default is accuracy.
        /// </summary>
        public MetricType metricType = MetricType.Accuracy;

        public double LearningRate = 0.1;

        /// <summary>
        /// The number of epochs to train.
        /// </summary>
        public int NumberOfEpochs { get; set; }

        /// <summary>
        /// The batch size to use.
        /// </summary>
        public int BatchSize { get; set; }

        /// <summary>
        /// The sequence length of the data.
        /// </summary>
        public int SequenceLength { get; set; } = 1;

        /// <summary>
        /// The neural network to train. 
        /// </summary>
        public CNTK.Function Model = null;

        /// <summary>
        /// The training curve with loss and accuracy values.
        /// </summary>
        public List<List<double>> TrainingCurves = null;

        // protected members
        protected CNTK.Variable features = null;
        protected CNTK.Variable labels = null;
        protected CNTK.Trainer trainer;
        protected CNTK.Evaluator evaluator;
        protected float[][] trainingFeatures;
        protected float[] trainingLabels;
        protected float[][] validationFeatures;
        protected float[] validationLabels;
        protected ReduceLROnPlateau scheduler;

        /// <summary>
        /// Construct a new instance of the training engine. 
        /// </summary>
        public TrainingEngine()
        {
            TrainingCurves = new List<List<double>>();
            TrainingCurves.Add(new List<double>());
            TrainingCurves.Add(new List<double>());
        }

        /// <summary>
        /// Set the training- and test features and labels.
        /// </summary>
        /// <param name="trainingFeatures">The training features to use</param>
        /// <param name="trainingLabels">The training labels to use</param>
        /// <param name="validationFeatures">The validation features to use</param>
        /// <param name="validationLabels">The validation labels to use</param>
        public void SetData(
            float[][] trainingFeatures, 
            float[] trainingLabels, 
            float[][] validationFeatures, 
            float[] validationLabels)
        {
            this.trainingFeatures = trainingFeatures;
            this.trainingLabels = trainingLabels;
            this.validationFeatures = validationFeatures;
            this.validationLabels = validationLabels;
        }

        /// <summary>
        /// Set up the feature variable.
        /// </summary>
        /// <returns>The feature variable to use.</returns>
        protected abstract CNTK.Variable CreateFeatureVariable();

        /// <summary>
        /// Set up the label variable.
        /// </summary>
        /// <returns>The label variable to use.</returns>
        protected abstract CNTK.Variable CreateLabelVariable();

        /// <summary>
        /// Create the model.
        /// </summary>
        /// <param name="features">The input feature to build the model on.</param>
        /// <returns>The completed model to use.</returns>
        protected abstract CNTK.Function CreateModel(CNTK.Variable features);

        /// <summary>
        /// Set up a custom loss function.
        /// </summary>
        /// <returns></returns>
        protected virtual CNTK.Function CustomLossFunction()
        {
            return null;
        }

        /// <summary>
        /// Check that the sequence length is okay.
        /// </summary>
        void AssertSequenceLength()
        {
            if (SequenceLength == 1) { return; }
            if (features.Shape.Dimensions.Count >= 2) { throw new NotImplementedException(); }
            if ((features.Shape.Dimensions.Count == 1) && (features.Shape.Dimensions[0] != 1)) { throw new NotImplementedException(); }
        }

        /// <summary>
        /// Train the model.
        /// </summary>
        /// <param name="threshold"></param>
        public void Train(double threshold = 0)
        {
            // create model and variables
            features = CreateFeatureVariable();
            labels = CreateLabelVariable();
            Model = CreateModel(features);
            AssertSequenceLength();

            // set up loss function
            CNTK.Function lossFunction = null;
            switch (lossFunctionType)
            {
                case LossFunctionType.BinaryCrossEntropy: lossFunction = CNTK.CNTKLib.BinaryCrossEntropy(Model, labels); break;
                case LossFunctionType.MSE: lossFunction = CNTK.CNTKLib.SquaredError(Model, labels); break;
                case LossFunctionType.CrossEntropyWithSoftmax: lossFunction = CNTK.CNTKLib.CrossEntropyWithSoftmax(Model, labels); break;
                case LossFunctionType.Custom: lossFunction = CustomLossFunction(); break;
            }

            // set up accuracy function
            CNTK.Function accuracy_function = null;
            switch (accuracyFunctionType)
            {
                case AccuracyFunctionType.SameAsLoss: accuracy_function = lossFunction; break;
                case AccuracyFunctionType.BinaryAccuracy: accuracy_function = NetUtil.BinaryAccuracy(Model, labels); break;
            }

            // set up an adam learner
            var learner = Model.GetAdamLearner(
                (LearningRate, (uint)BatchSize),   // remove batch_size?
                (0.9, (uint)BatchSize),  // remove batch_size?
                unitGain: false);

            // set up trainer
            trainer = CNTK.CNTKLib.CreateTrainer(Model, lossFunction, accuracy_function, new CNTK.LearnerVector() { learner });

            // set up a scheduler to tweak the learning rate
            scheduler = new ReduceLROnPlateau(learner, LearningRate);

            // set up an evaluator
            if (validationFeatures != null)
            {
                evaluator = CNTK.CNTKLib.CreateEvaluator(accuracy_function);
            }

            // write the model summary
            Console.WriteLine("  Model architecture:");
            Console.WriteLine(Model.ToSummary());

            // clear the training curves
            TrainingCurves[0].Clear();
            TrainingCurves[1].Clear();

            // train for a certain number of epochs
            for (int epoch = 0; epoch < NumberOfEpochs; epoch++)
            {
                var epoch_start_time = DateTime.Now;

                // train and evaluate the model
                var epoch_training_metric = TrainBatches();
                var epoch_validation_accuracy = EvaluateBatches();

                // add to training curve
                TrainingCurves[0].Add(epoch_training_metric);
                TrainingCurves[1].Add(epoch_validation_accuracy);

                // write current loss and accuracy
                var elapsedTime = DateTime.Now.Subtract(epoch_start_time);
                if (metricType == MetricType.Accuracy)
                {
                    Console.WriteLine($"Epoch {epoch + 1:D2}/{NumberOfEpochs}, Elapsed time: {elapsedTime.TotalSeconds:F3} seconds. " +
                      $"Training Accuracy: {epoch_training_metric:F3}. Validation Accuracy: {epoch_validation_accuracy:F3}.");
                }
                else
                {
                    Console.WriteLine($"Epoch {epoch + 1:D2}/{NumberOfEpochs}, Elapsed time: {elapsedTime.TotalSeconds:F3} seconds, Training Loss: {epoch_training_metric:F3}");
                }

                // abort training if scheduler says so
                if (scheduler.Update(epoch_training_metric))
                {
                    break;
                }
                if ((threshold != 0) && (epoch_training_metric < threshold))
                {
                    break;
                }
            }
        }

        /// <summary>
        /// Evaluate the model.
        /// </summary>
        /// <param name="featureData">The data to evaluate the model on</param>
        /// <param name="model">The model to use (defaults to trained model)</param>
        /// <returns>The output of the model</returns>
        public IList<IList<float>> Evaluate(float[][] featureData, CNTK.Function model = null)
        {
            // get the current model
            if (model == null)
            {
                model = this.Model;
            }

            // get the current batch
            var featureBatch = (SequenceLength == 1) ?
              features.GetBatch(featureData, 0, featureData.Length) :
              features.GetSequenceBatch(SequenceLength, featureData, 0, featureData.Length);

            // evaluate the model
            var inputs = new Dictionary<CNTK.Variable, CNTK.Value>() { { features, featureBatch } };
            var outputs = new Dictionary<CNTK.Variable, CNTK.Value>() { { model.Output, null } };
            model.Evaluate(inputs, outputs, NetUtil.CurrentDevice);

            // return result
            var result = outputs[model.Output];
            var outputData = result.GetDenseData<float>(model.Output);
            return outputData;
        }

        /// <summary>
        /// Evaluate on a collection of batches.
        /// </summary>
        /// <returns>The final value of the metric after evaluation.</returns>
        double EvaluateBatches()
        {
            // return if we have no evaluator
            if (evaluator == null)
            {
                return 0.0;
            }

            // loop through each batch of data
            var metric = 0.0;
            validationFeatures.Batch(
                BatchSize,
                (data, begin, end) =>
                {
                    // get current batch for testing
                    var featureBatch = (SequenceLength == 1) ?
                        features.GetBatch(validationFeatures, begin, end) :
                        features.GetSequenceBatch(SequenceLength, validationFeatures, begin, end);
                    var labelBatch = labels.GetBatch(validationLabels, begin, end);

                    // return if we have no validation features
                    if (validationFeatures.Length == 0)
                    {
                        return;
                    }

                    // test the network on the batch
                    var minibatch_metric = evaluator.TestBatch(
                        new[]
                        {
                                (features, featureBatch),
                                (labels, labelBatch)
                        });

                    metric += minibatch_metric * (end - begin);
                    featureBatch.Erase();
                    labelBatch.Erase();
                });

            // return the average metric value
            metric /= validationFeatures.Length;
            return metric;
        }

        /// <summary>
        /// Train on a collection of batches.
        /// </summary>
        /// <returns>The final value of the metric after training.</returns>
        double TrainBatches()
        {
            // loop through each batch of data
            var metric = 0.0;
            trainingFeatures.Index().Shuffle().Batch(
                BatchSize,
                (indices, begin, end) =>
                {
                    // get current batch for training
                    var featureBatch = (SequenceLength == 1) ?
                      features.GetBatch(trainingFeatures, indices, begin, end) :
                      features.GetSequenceBatch(SequenceLength, trainingFeatures, indices, begin, end);
                    var labelBatch = labels.GetBatch(trainingLabels, indices, begin, end);

                    // train the network on the batch
                    bool isSweepEndInArguments = (end == indices.Length);
                    trainer.TrainBatch(
                        new[]
                        {
                                (features, featureBatch),
                                (labels, labelBatch)
                        },
                        isSweepEndInArguments
                    );

                    // update metric
                    var minibatch_metric = (metricType == MetricType.Loss) ? trainer.PreviousMinibatchLossAverage() : trainer.PreviousMinibatchEvaluationAverage();
                    metric += minibatch_metric * (end - begin);

                    // erase batches
                    featureBatch.Erase();
                    labelBatch.Erase();
                });

            // return average of metric
            metric /= trainingFeatures.GetLength(0);
            return metric;
        }
    }

}

