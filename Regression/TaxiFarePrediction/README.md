# Assignment: Predict taxi fares in New York

In this assignment you're going to build an app that can predict taxi fares in New York.

The first thing you'll need is a data file with transcripts of New York taxi rides. The [NYC Taxi & Limousine Commission](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) provides yearly TLC Trip Record Data files which have exactly what you need.

Download the [Yellow Taxi Trip Records from December 2018](#) and save the file as **yellow_tripdata_2018-12_small.csv**. 

This is a CSV file with 999 records that looks like this:
￼
![Data File](./assets/data.png)

There are a lot of columns with interesting information in this data file, but you will only train on the following:

* Column 0: The data provider vendor ID
* Column 3: Number of passengers
* Column 4: Trip distance
* Column 5: The rate code (standard, JFK, Newark, …)
* Column 9: Payment type (credit card, cash, …)
* Column 10: Fare amount

You are going to build a linear regression model in C# that will use columns 0, 3, 4, 5, and 9 as input, and use them to predict the taxi fare for every trip. Then you’ll compare the predicted fares with the actual taxi fares in column 10, and evaluate the accuracy of your model.

In these assignments you will not be using the code in Github. Instead, you'll be building all the applications 100% from scratch. So please make sure to create a new folder somewhere to hold all of your assignments.

Now please open a console or Powershell window. You are going to create a new subfolder for this assignment and set up a blank console application:

```bash
$ dotnet new console -o TaxiFarePrediction
$ cd TaxiFarePrediction
```

Also make sure to copy the dataset file **yellow_tripdata_2018-12_small.csv** into this folder because the code you're going to type next will expect it here.  

Now install the following packages

```bash
$ dotnet add package Microsoft.ML
$ dotnet add package CNTK.GPU
```

**Microsoft.ML** is the Microsoft machine learning package. We will use to load and process the data from the dataset. 

The **CNTK.GPU** library is Microsoft's Cognitive Toolkit that can train and run deep neural networks. The library will use your GPU and you'll need an NVidia GPU and Cuda graphics drivers for this to work. 

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

    // the rest of the code goes here...
}
```

The **TaxiTrip** class holds one single taxi trip. Note how each field is tagged with a **LoadColumn** attribute that tells the CSV data loading code which column to import data from.

We also have a **GetFeatures** method that returns the vendor id, rate code, passenger count, trip distance, and payment type of a taxi trip.

And there's a **GetLabel** method that return the fare amount of the taxi trip in dollars.

The features are the taxi trip attributes that we will use to train a linear regression model on, and the label is the output variable that we're trying to predict. So here we're training on five columns in the dataset to predict the fare amount. 

Now it's time to start writing the main program method:

```csharp
/// <summary>
/// The program class.
/// </summary>
class Program
{
    // file paths to data files
    static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "yellow_tripdata_2018-12_small.csv");

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

        // the rest of the code goes here...
    }
}
```

This code sets up a **TextLoader** to load the CSV data into memory. Note that all column data types are set to float. 

With the TextLoader all set up, a single call to **Load** is sufficient to load the entire data file in memory.

Finally we call **CreateEnumerable** to convert the data to an enumeration of **TaxiTrip** instances. So now we have the training data available in the **training** variable as an enumeration of **TaxiTrip** instances.

But the CNTK library we're using for linear regression cannot train on an enumeration of class instances. It requires a **float[][]** for features and **float[]** for labels.

So we need to set up two float arrays:

```csharp
// set up data arrays
var training_data = training.Select(v => v.GetFeatures()).ToArray();
var training_labels = training.Select(v => v.GetLabel()).ToArray();

// the rest of the code goes here...
```

These LINQ expressions set up two arrays containing the feature and label training data.  

Now we need to tell CNTK what shape the input data has that we'll train the model on, and what shape the output predictions will have.

Here we are training on taxi trips consisting of five float values, and we're trying to predict the fare amount which is a single float value:

```csharp
// build features and labels
var features = NetUtil.Var(new int[] { 5 }, DataType.Float);
var labels = NetUtil.Var(new int[] { 1 }, DataType.Float);

// the rest of the code goes here...
```

Note the first **Var** method which tells CNTK that our model will use an array of 5 float values as input. This shape matches the 5 values returned by the **TaxiTrip.GetFeatures** method. 

And the second **Var** method tells CNTK that we want our model to output predictions as a single float value. This shape matches the single value returned by the **TaxiTrip.GetLabel** method.

Our next step is to build the linear regression model that will make the predictions. Setting up a linear regression model in CNTK is very easy. We can use the **Dense** method to set it up:

```csharp
// build the network
var network = features
    .Dense(1)
    .ToNetwork();
Console.WriteLine("Model architecture:");
Console.WriteLine(network.ToSummary());

// the rest of the code goes here...
```

Linear regression corresponds to a neural network with one single output node. Note the '1' value we specified as the argument in our call to **Dense**, this is the number of nodes we want to have in our network.

This single call sets up a deep neural network with one single feed-forward layer containing one single node. This is mathematically the same as a linear regression operation. 

Finally we use the **ToSummary** method to output a description of the model to the console.

Now we need to decide which loss function to use to train the model, and how we are going to track the prediction error of the model during each training epoch. 

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

We're almost ready to train. Our final step is to set up a trainer for calculating the training loss and the error during each epoch:

```csharp
// set up a trainer and an evaluator
var trainer = network.GetTrainer(learner, lossFunc, errorFunc);

// train the model
Console.WriteLine("Epoch\tTrain\tTrain");
Console.WriteLine("\tLoss\tError");
Console.WriteLine("-----------------------");

// the rest of the code goes here...
```

The **GetTrainer** method sets up a trainer which will track the loss and the error during training. 

Now we're finally ready to start training the model!

Add the following code:

```csharp
var maxEpochs = 50;
var batchSize = 32;
var loss = new double[maxEpochs];
var trainingError = new double[maxEpochs];
var batchCount = 0;
for (int epoch = 0; epoch < maxEpochs; epoch++)
{
    // training and testing code goes here...
}

// show final results
var finalError = trainingError[maxEpochs-1];
Console.WriteLine();
Console.WriteLine($"Final MAE: {finalError:0.00}");

// plotting code goes here...
```

We're training the network for 50 epochs using a batch size of 32. During training we'll track the loss and errors in the **loss** and **trainingError** arrays.

Once training is done, we show the final training error on the console. This is the final average fare amount prediction error expressed in dollars.

Here's the code to train the model:

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
Console.WriteLine($"{epoch}\t{loss[epoch]:F3}\t{trainingError[epoch]:F3}");
```

The **Index().Shuffle().Batch()** sequence randomizes the data and splits it up in a collection of 32-record batches. The second argument to **Batch()** is a function that will be called for every batch.

Inside the batch function we call **GetBatch** twice to get a feature batch and a corresponding label batch. Then we call **TrainBatch** to train the neural network on these two batches of training data.

The **TrainBatch** method returns the loss and error, but only for training on the 32-record batch. So we simply add up all these values and divide them by the number of batches in the dataset. That gives us the average loss and error for the predictions on the training partition during the current epoch, and we report this to the console.

We're now ready to build the app, so this is a good moment to save your work ;) 

Go to the CNTKUtil folder and type the following:

```bash
$ dotnet build -o bin/Debug/netcoreapp3.0 -p:Platform=x64
```

This will build the CNKTUtil project. Note how we're specifying the x64 platform because the CNTK library requires a 64-bit build. 

Now go to the TaxiFarePrediction folder and type:

```bash
$ dotnet build -o bin/Debug/netcoreapp3.0 -p:Platform=x64
```

This will build your app. Note how we're again specifying the x64 platform.

Now run the app:

```bash
$ dotnet run
```

The app will create the regression model, load the dataset, train the network on the data, and report the final MAE value for the fully trained model. 

What is your final MAE value? Are you happy with the result?

Try to improve the predictions by changing the number of epochs, the batch size, or the learner parameters.

Did the changes help? What is the best MAE you can achieve?

Now I have a confession to make: 

You've been working with a tiny subset that only contains about 0.01% of of the full taxi data. The complete dataset actually has 8,173,233 records!

Download the complete [Yellow Taxi Trip Records from December 2018](https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2018-12.csv) and save the file as **yellow_tripdata_2018-12.csv** in your project folder.

Now change your **dataPath** field like this:

```csharp
// file paths to data files
static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "yellow_tripdata_2018-12.csv");

```

Run your app again. What is your final MAE now? Can you improve your results by tweaking the batch size and the number of epochs? What's the best MAE you can achieve?

Why do you think you are getting different results for the small and the full data file? Does the difference make sense to you?

Post your results in our support group!