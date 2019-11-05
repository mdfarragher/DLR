# Assignment: Predict heart disease risk

In this assignment you're going to build an app that can predict the heart disease risk in a group of patients.

The first thing you will need for your app is a data file with patients, their medical info, and their heart disease risk assessment. We're going to use the famous [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) which has real-life data from 303 patients.

Download the [Processed Cleveland Data](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data) file and save it as **processed.cleveland.data.csv**.

The data file looks like this:

![Processed Cleveland Data](./assets/data.png)

It’s a CSV file with 14 columns of information:

* Age
* Sex: 1 = male, 0 = female
* Chest Pain Type: 1 = typical angina, 2 = atypical angina , 3 = non-anginal pain, 4 = asymptomatic
* Resting blood pressure in mm Hg on admission to the hospital
* Serum cholesterol in mg/dl
* Fasting blood sugar > 120 mg/dl: 1 = true; 0 = false
* Resting EKG results: 0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes’ criteria
* Maximum heart rate achieved
* Exercise induced angina: 1 = yes; 0 = no
* ST depression induced by exercise relative to rest
* Slope of the peak exercise ST segment: 1 = up-sloping, 2 = flat, 3 = down-sloping
* Number of major vessels (0–3) colored by fluoroscopy
* Thallium heart scan results: 3 = normal, 6 = fixed defect, 7 = reversible defect
* Diagnosis of heart disease: 0 = normal risk, 1-4 = elevated risk

The first 13 columns are patient diagnostic information, and the last column is the diagnosis: 0 means a healthy patient, and values 1-4 mean an elevated risk of heart disease.

You are going to build a binary classification network that reads in patient information and then makes a prediction for the heart disease risk.

Let’s get started. You need to build a new application from scratch by opening a terminal and creating a new NET Core console project:

```bash
$ dotnet new console -o HeartDisease
$ cd HeartDisease
```

Also make sure to copy the dataset file **processed.cleveland.data.csv** into this folder because the code you're going to type next will expect it here.  

Now install the following packages

```bash
$ dotnet add package Microsoft.ML
$ dotnet add package CNTK.GPU
$ dotnet add package XPlot.Plotly
$ dotnet add package Fsharp.Core
```

**Microsoft.ML** is the Microsoft machine learning package. We will use to load and process the data from the dataset. The **CNTK.GPU** library is Microsoft's Cognitive Toolkit that can train and run deep neural networks. And **Xplot.Plotly** is an awesome plotting library based on Plotly. The library is designed for F# so we also need to pull in the **Fsharp.Core** library. 

The **CNTK.GPU** package will train and run deep neural networks using your GPU. You'll need an NVidia GPU and Cuda graphics drivers for this to work. 

If you don't have an NVidia GPU or suitable drivers, the library will fall back and use the CPU instead. This will work but training neural networks will take significantly longer.

CNTK is a low-level tensor library for building, training, and running deep neural networks. The code to build deep neural network can get a bit verbose, so I've developed a little wrapper called **CNTKUtil** that will help you write code faster. 

Please [download the CNTKUtil files](https://github.com/mdfarragher/DLR/tree/master/CNTKUtil) in a new **CNTKUtil** folder at the same level as your project folder.

Then make sure you're in the console project folder and crearte a project reference like this:

```bash
$ dotnet add reference ..\CNTKUtil\CNTKUtil.csproj
```

Now you are ready to start writing code. Edit the Program.cs file with Visual Studio Code and add the following code:

```csharp
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

    // the rest of the code goes here...
}
```

The **HeartData** class holds all the data for one single patient. Note how each field is tagged with a **LoadColumn** attribute that will tell the CSV data loading code from which column to import the data.

We also have a **GetFeatures** method that returns a subset of the data columns: the age, cholesterol level, fasting blood sugar, maximum heart rate, exercise induced angina, ST depression, and slope value of the patient.

And there's a **GetLabel** method that returns 1 if the patient is a heart disease risk and 0 if the patient is not.

The features are the patient attributes that we will use to train the neural network on, and the label is the output variable that we're trying to predict. So here we're training on 7 patient attributes in the dataset to predict the heart disease risk. 

Now it's time to start writing the main program method:

```csharp
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

        // the rest of the code goes here...
    }
}
```

When working with the ML.NET library we always need to set up a machine learning context represented by the **MLContext** class.

The code calls the **LoadFromTextFile** method to load the CSV data in memory. Note the **HeartData** type argument that tells the method which class to use to load the data.

We then use **TrainTestSplit** to split the data in a training partition containing 60% of the data and a testing partition containing 40% of the data.

Note that we're deviating from the usual 80-20 split here. This is because the data file is extremely small, and so 20% of the data is simply not enough to test the neural network on. 

Finally we call **CreateEnumerable** to convert the two partitions to an enumeration of **HeartData** instances. So now we have the training data in **training** and the testing data in **testing**. Both are enumerations of **HeartData** instances.

But CNTK can't train on an enumeration of class instances. It requires a **float[][]** for features and **float[]** for labels.

So we need to set up four float arrays:

```csharp
// set up data arrays
var training_data = training.Select(v => v.GetFeatures()).ToArray();
var training_labels = training.Select(v => v.GetLabel()).ToArray();
var testing_data = testing.Select(v => v.GetFeatures()).ToArray();
var testing_labels = testing.Select(v => v.GetLabel()).ToArray();

// the rest of the code goes here...
```

These LINQ expressions set up four arrays containing the feature and label data for the training and testing partitions.  

Now we need to tell CNTK what shape the input data has that we'll train the neural network on, and what shape the output data of the neural network will have: 

```csharp
// build features and labels
var features = NetUtil.Var(new int[] { 7 }, DataType.Float);
var labels = NetUtil.Var(new int[] { 1 }, DataType.Float);

// the rest of the code goes here...
```

Note the first **Var** method which tells CNTK that our neural network will use a 1-dimensional tensor of 7 float values as input. This shape matches the 7 values returned by the **HeartData.GetFeatures** method. 

And the second **Var** method tells CNTK that we want our neural network to output a single float value. This shape matches the single value returned by the **HeartData.GetLabel** method.

Our next step is to design the neural network. 

We will use a deep neural network with a 16-node input layer, a 128-node hidden layer, and a single-node output layer. We'll use the **ReLU** activation function for the input and hidden layers, and **Sigmoid** activation for the output layer. 

Remember: the sigmoid function forces the output to a range of 0..1 which means we can treat it as a binary classification probability. So we can turn any regression network into a binary classification network by simply adding the sigmoid activation function to the output layer.

Here's how to build this neural network:

```csharp
// build the network
var network = features
    .Dense(16, CNTKLib.ReLU)
    .Dense(128, CNTKLib.ReLU)
    .Dense(1, CNTKLib.Sigmoid)
    .ToNetwork();
Console.WriteLine("Model architecture:");
Console.WriteLine(network.ToSummary());

// the rest of the code goes here...
```

Each **Dense** call adds a new dense feedforward layer to the network. We're stacking two layers, both using **ReLU** activation, and then add a final layer with a single node using **Sigmoid** activation.

Then we use the **ToSummary** method to output a description of the architecture of the neural network to the console.

Now we need to decide which loss function to use to train the neural network, and how we are going to track the prediction error of the network during each training epoch. 

For this assignment we'll use **BinaryCrossEntropy** as the loss function because it's the standard metric for measuring binary classification loss. 

We'll track the error with the **BinaryClassificationError** metric. This is the number of times (expressed as a percentage) that the model predictions are wrong. An error of 0 means the predictions are correct all the time, and an error of 1 means the predictions are wrong all the time. 

```csharp
// set up the loss function and the classification error function
var lossFunc = CNTKLib.BinaryCrossEntropy(network.Output, labels);
var errorFunc = NetUtil.BinaryClassificationError(network.Output, labels);

// the rest of the code goes here...
```

Next we need to decide which algorithm to use to train the neural network. There are many possible algorithms derived from Gradient Descent that we can use here.

For this assignment we're going to use the **AdamLearner**. You can learn more about the Adam algorithm here: [https://machinelearningmastery.com/adam...](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

```csharp
// set up a learner
var learner = network.GetAdamLearner(
    learningRateSchedule: (0.001, 1),
    momentumSchedule: (0.9, 1),
    unitGain: true);

// the rest of the code goes here...
```

These configuration values are a good starting point for many machine learning scenarios, but you can tweak them if you like to try and improve the quality of your predictions.

We're almost ready to train. Our final step is to set up a trainer and an evaluator for calculating the loss and the error during each training epoch:

```csharp
// set up a trainer and an evaluator
var trainer = network.GetTrainer(learner, lossFunc, errorFunc);
var evaluator = network.GetEvaluator(errorFunc);

// train the model
Console.WriteLine("Epoch\tTrain\tTrain\tTest");
Console.WriteLine("\tLoss\tError\tError");
Console.WriteLine("-----------------------------");

// the rest of the code goes here...
```

The **GetTrainer** method sets up a trainer which will track the loss and the error for the training partition. And **GetEvaluator** will set up an evaluator that tracks the error in the test partition. 

Now we're finally ready to start training the neural network!

Add the following code:

```csharp
var maxEpochs = 100;
var batchSize = 1;
var loss = new double[maxEpochs];
var trainingError = new double[maxEpochs];
var testingError = new double[maxEpochs];
var batchCount = 0;
for (int epoch = 0; epoch < maxEpochs; epoch++)
{
    // training and testing code goes here...
}

// show final results
var finalError = testingError[maxEpochs-1];
Console.WriteLine();
Console.WriteLine($"Final test error: {finalError:0.00}");
Console.WriteLine($"Final test accuracy: {1 - finalError:0.00}");

// plotting code goes here...
```

We're training the network for 100 epochs using a batch size of 1. During training we'll track the loss and errors in the **loss**, **trainingError** and **testingError** arrays.

A batch size of one means the neural network is trained on each individual record in the dataset. This produces the best possible prediction accuracy, but also takes the longest to train. But because the dataset for this assignment is so tiny, training with batches of one record is more than fast enough here.

Once training is done, we show the final testing error on the console. This is the percentage of mistakes the network makes when predicting heart disease risk. 

Note that the error and the accuracy are related: accuracy = 1 - error. So we also report the final accuracy of the neural network. 

Here's the code to train the neural network. Put this inside the for loop:

```csharp
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

// testing code goes here...
```

The **Index().Shuffle().Batch()** sequence randomizes the data and splits it up in a collection of 1-record batches. The second argument to **Batch()** is a function that will be called for every batch.

Inside the batch function we call **GetBatch** twice to get a feature batch and a corresponding label batch. Then we call **TrainBatch** to train the neural network on these two batches of training data.

The **TrainBatch** method returns the loss and error, but only for training on the 1-record batch. So we simply add up all these values and divide them by the number of batches in the dataset. That gives us the average loss and error for the predictions on the training partition during the current epoch, and we report this to the console.

So now we know the training loss and error for one single training epoch. The next step is to test the network by making predictions about the data in the testing partition and calculate the testing error.

Put this code inside the epoch loop and right below the training code:

```csharp
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
```

We don't need to shuffle the data for testing, so now we can call **Batch** directly. Again we're calling **GetBatch** to get feature and label batches, but note that we're now providing the **testing_data** and **testing_labels** arrays. 

We call **TestBatch** to test the neural network on the 1-record test batch. The method returns the error for the batch, and we again add up the errors for each batch and divide by the number of batches. 

That gives us the average error in the neural network predictions on the test partition for this epoch. 

After training completes, the training and testing errors for each epoch will be available in the **trainingError** and **testingError** arrays. Let's use XPlot to create a nice plot of the two error curves so we can check for overfitting:

```csharp
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
```

This code creates a **Plot** with two **Scatter** graphs. The first one plots the **trainingError** values and the second one plots the **testingError** values. 

Finally we use **File.WriteAllText** to write the plot to disk as a HTML file.

We're now ready to build the app, so this is a good moment to save your work ;) 

Go to the CNTKUtil folder and type the following:

```bash
$ dotnet build -o bin/Debug/netcoreapp3.0 -p:Platform=x64
```

This will build the CNKTUtil project. Note how we're specifying the x64 platform because the CNTK library requires a 64-bit build. 

Now go to the HeartDisease folder and type:

```bash
$ dotnet build -o bin/Debug/netcoreapp3.0 -p:Platform=x64
```

This will build your app. Note how we're again specifying the x64 platform.

Now run the app:

```bash
$ dotnet run
```

The app will create the neural network, load the dataset, train the network on the data, and create a plot of the training and testing errors for each epoch. 

The plot is written to disk in a new file called chart.html. Open the file now and take a look at the training and testing curves.

What are your final classification errors on training and testing? What is the final testing accuracy? And what do the curves look like? Is the neural network overfitting?

Do you think this model is good at predicting heart disease?

Try to improve the neural network by changing the network architecture. You can add more nodes or extra layers. You can also changing the number of epochs, the batch size, or the learner parameters. Or you can try to train on different patient attributes.

Did the changes help? What is the best accuracy you can achieve?

You may notice that sometimes the neural network stalls and won't train at all, and all loss and error values are constant during training. This happens when the dataset is too complex to train on. The gradient descent learner cannot find any workable solution at all, and so during each training epoch the neural network hardly changes.

When stalling  happens, you need to simplify your model. Remove some input attributes, reduce the number of nodes in a layer, or remove a layer. Keep simplifying until your network starts training again.

Did you see your network stalling? When did this happen? How did you fix it?

Post your results in our support group.
