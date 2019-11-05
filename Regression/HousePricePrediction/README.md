# Assignment: Predict house prices in California

In this assignment, you're going to build an app that trains a deep neural network on a dataset of house prices in the state of California. 

So the first thing you'll need is the data file with house prices. The 1990 California census has exactly what we need. 

Download the [California 1990 housing census](https://github.com/mdfarragher/DLR/blob/master/Regression/HousePricePrediction/california_housing.csv) and save it as **california_housing.csv**. 

This is a CSV file with 17,000 records that looks like this:

￼
![Data File](./assets/data.png)

The file contains information on 17k housing blocks all over the state of California:

* Column 1: The longitude of the housing block
* Column 2: The latitude of the housing block
* Column 3: The median age of all the houses in the block
* Column 4: The total number of rooms in all houses in the block
* Column 5: The total number of bedrooms in all houses in the block
* Column 6: The total number of people living in all houses in the block
* Column 7: The total number of households in all houses in the block
* Column 8: The median income of all people living in all houses in the block
* Column 9: The median house value for all houses in the block

We can use this data to train a deep neural network to predict the value of any house in and outside the state of California. 

Let's get started. 

In these assignments you will not be using the code in Github. Instead, you'll be building all the applications 100% from scratch. So please make sure to create a new folder somewhere to hold all of your assignments.

Now please open a console or Powershell window. You are going to create a new subfolder for this assignment and set up a blank console application:

```bash
$ dotnet new console -o HousePricePrediction
$ cd HousePricePrediction
```

Also make sure to copy the dataset file **california_housing.csv** into this folder because the code you're going to type next will expect it here.  

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

    // the rest of the code goes here...
}
```

The **HouseBlockData** class holds all the data for one single housing block. Note how each field is tagged with a **LoadColumn** attribute that will tell the CSV data loading code from which column to import the data.

We also have a **GetFeatures** method that returns the longitude, latitude, median age, total number of rooms, total number of bedrooms, total population, number of households, and median income level of a housing block.

And there's a **GetLabel** method that return the median house value in thousands of dollars.

The features are the house attributes that we will use to train the neural network on, and the label is the output variable that we're trying to predict. So here we're training on every column in the dataset to predict the median house value. 

Now it's time to start writing the main program method:

```csharp
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

        // check the current device for running neural networks
        Console.WriteLine($"Using device: {NetUtil.CurrentDevice.AsString()}");

        // the rest of the code goes here...
    }
}
```

When working with the ML.NET library we always need to set up a machine learning context represented by the **MLContext** class.

Also note that we're reporting the **CurrentDevice**, this is the hardware device that CNTK will use to train and run neural networks. If the app reports a GPU device, it means you have set up your graphics device driver and the CUDA library correctly on your system. 

Now let's use ML.NET to load the dataset:

```csharp
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

// the rest of the code goes here...
```

This code calls the **LoadFromTextFile** method to load the CSV data in memory. Note the **HouseBlockData** type argument that tells the method which class to use to load the data.

We then use **TrainTestSplit** to split the data in a training partition containing 80% of the data and a testing partition containing 20% of the data.

Finally we call **CreateEnumerable** to convert the two partitions to an enumeration of **HouseBlockData** instances. So now we have the training data in **training** and the testing data in **testing**. Both are enumerations of **HouseBlockData** instances.

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
var features = NetUtil.Var(new int[] { 8 }, DataType.Float);
var labels = NetUtil.Var(new int[] { 1 }, DataType.Float);

// the rest of the code goes here...
```

Note the first **Var** method which tells CNTK that our neural network will use a 1-dimensional tensor of 8 float values as input. This shape matches the 8 values returned by the **HouseBlockData.GetFeatures** method. 

And the second **Var** method tells CNTK that we want our neural network to output a single float value. This shape matches the single value returned by the **HouseBlockData.GetLabel** method.

Our next step is to design the neural network. 

We will use the following neural network to predict house prices:

![Neural network](./assets/network.png)

This is a deep neural network with an 8-node input layer, an 8-node hidden layer, and a single-node output layer. We'll use the **ReLU** activation function everywhere. 

Here's how to build this neural network:

```csharp
// build the network
var network = features
    .Dense(8, CNTKLib.ReLU)
    .Dense(8, CNTKLib.ReLU)
    .Dense(1)
    .ToNetwork();
Console.WriteLine("Model architecture:");
Console.WriteLine(network.ToSummary());

// the rest of the code goes here...
```

Each **Dense** call adds a new dense feedforward layer to the network. We're stacking two layers with 8 nodes each, both using **ReLU** activation, and then add a final layer with only a single node.

Then we use the **ToSummary** method to output a description of the architecture of the neural network to the console.

Now we need to decide which loss function to use to train the neural network, and how we are going to track the prediction error of the network during each training epoch. 

There's nothing stopping us from using the same function for both loss and error, but often it's nice to use separate metrics for training and error reporting. 

For this assignment we'll use **MSE** as the loss function because it's the standard metric for measuring regression loss. But we'll track the error with the **MAE** metric. The nice thing about MAE is that it expresses the average prediction error in dollars. 

```csharp
// set up the loss function and the classification error function
var lossFunc = NetUtil.MeanSquaredError(network.Output, labels);
var errorFunc = NetUtil.MeanAbsoluteError(network.Output, labels);

// the rest of the code goes here...
```

Next we need to decide which algorithm to use to train the neural network. There are many possible algorithms derived from Gradient Descent that we can use here.

For this assignment we're going to use the **AdamLearner**. You can learn more about the Adam algorithm here: [https://machinelearningmastery.com/adam...](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

```csharp
// set up a learner
var learner = network.GetAdamLearner(
    learningRateSchedule: (0.001, 1),
    momentumSchedule: (0.9, 1),
    unitGain: false);

// the rest of the code goes here...
```

These configuration values are a good starting point for many machine learning scenarios, but you can tweak them if you like to try and improve the quality of your predictions.

We're almost ready to train. Our final step is to set up a trainer and an evaluator for calculating the loss and the error during each training epoch:

```csharp
// set up a trainer and an evaluator
var trainer = network.GetTrainer(learner, lossFunc, errorFunc);
var evaluator = network.GetEvaluator(errorFunc);

// train the model
Console.WriteLine("Epoch\tTrain\t\tTrain\tTest");
Console.WriteLine("\tLoss\t\tError\tError");
Console.WriteLine("--------------------------------------");

// the rest of the code goes here...
```

The **GetTrainer** method sets up a trainer which will track the loss and the error for the training partition. And **GetEvaluator** will set up an evaluator that tracks the error in the test partition. 

Now we're finally ready to start training the neural network!

Add the following code:

```csharp
var maxEpochs = 50;
var batchSize = 16;
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
Console.WriteLine($"Final test MAE: {finalError:0.00}");

// plotting code goes here...
```

We're training the network for 50 epochs using a batch size of 16. During training we'll track the loss and errors in the **loss**, **trainingError** and **testingError** arrays.

Once training is done, we show the final testing error on the console. This is the final average house price prediction error expressed in thousands of dollars.

Here's the code to train the neural network:

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

The **Index().Shuffle().Batch()** sequence randomizes the data and splits it up in a collection of 16-record batches. The second argument to **Batch()** is a function that will be called for every batch.

Inside the batch function we call **GetBatch** twice to get a feature batch and a corresponding label batch. Then we call **TrainBatch** to train the neural network on these two batches of training data.

The **TrainBatch** method returns the loss and error, but only for training on the 16-record batch. So we simply add up all these values and divide them by the number of batches in the dataset. That gives us the average loss and error for the predictions on the training partition during the current epoch, and we report this to the console.

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

We call **TestBatch** to test the neural network on the 16-record test batch. The method returns the error for the batch, and we again add up the errors for each batch and divide by the number of batches. 

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
chart.WithYTitle("Mean absolute error (MAE)");
chart.WithTitle("California House Training");

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

Now go to the HousePricePrediction folder and type:

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

What are your final MAE values on training and testing? And what do the curves look like? Is the neural network overfitting?

Try to improve the neural network by changing the network architecture. You can add more nodes or extra layers. You can also changing the number of epochs, the batch size, or the learner parameters.

Did the changes help? What is the best MAE you can achieve?

Post your results in our support group.
