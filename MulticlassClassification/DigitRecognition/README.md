# Assignment: Recognize handwritten digits

In this article, You are going to build an app that recognizes handwritten digits from the famous MNIST machine learning dataset:

![MNIST digits](./assets/mnist.png)

Your app must read these images of handwritten digits and correctly predict which digit is visible in each image.

This may seem like an easy challenge, but look at this:

![Difficult MNIST digits](./assets/mnist_hard.png)

These are a couple of digits from the dataset. Are you able to identify each one? It probably won’t surprise you to hear that the human error rate on this exercise is around 2.5%.

The first thing you will need for your app is a data file with images of handwritten digits. We will not use the original MNIST data because it's stored in a nonstandard binary format.

Instead, we'll use these excellent [CSV files](https://www.kaggle.com/oddrationale/mnist-in-csv/) prepared by Daniel Dato on Kaggle.

Create a Kaggle account if you don't have one yet, then download **mnist_train.csv** and **mnist_test.csv** and save them in your project folder.

There are 60,000 images in the training file and 10,000 in the test file. Each image is monochrome and resized to 28x28 pixels.

The training file looks like this:

![Data file](./assets/datafile.png)

It’s a CSV file with 785 columns:

* The first column contains the label. It tells us which one of the 10 possible digits is visible in the image.
* The next 784 columns are the pixel intensity values (0..255) for each pixel in the image, counting from left to right and top to bottom.

You are going to build a multiclass classification network that reads in all 785 columns, and then makes a prediction for each digit in the dataset.

Let’s get started. You need to build a new application from scratch by opening a terminal and creating a new NET Core console project:

```bash
$ dotnet new console -o Mnist
$ cd Mnist
```

Also make sure to copy the dataset files **mnist_train.csv** and **mnist_test.csv** into this folder because the code you're going to type next will expect it here.  

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

namespace Mnist
{
    /// <summary>
    /// The Digit class represents one mnist digit.
    /// </summary>
    class Digit
    {
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

    // the rest of the code goes here...
}
```

The **Digit** class holds one single MNIST digit image. Note how the **PixelValues** field is tagged with a **VectorType** attribute. This tells ML.NET to combine the 784 individual pixel columns into a single vector value. Also note the **LoadColumn** attribute that tells ML.NET to load the first CSV column into the **Number** field and all subsequent columns into the **PixelValues** field.

We also have a **GetFeatures** method that returns the pixel values as a float array, and a **GetLabel** method that returns a one-hot encoded float array of the digit value. For each digit image only a single element in the float array will contain a 1 value to indicate the numerical value of that digit. 

The features are the pixels in the image that we will use to train the neural network on, and the label is the digit value that we're trying to predict. So here we're training on all 784 digit pixels in the dataset to predict the value of the digit. 

Now it's time to start writing the main program method:

```csharp
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


        // the rest of the code goes here...
    }
}
```

This code uses the **LoadFromTextFile** method to load the CSV data directly into memory. Note the **columnDef** variable that instructs ML.NET to load CSV columns 1..784 into the **PixelValues** column, and CSV column 0 into the **Number** column.

Finally we call **CreateEnumerable** to convert the training and test data to an enumeration of **Digit** instances. So now we have the training data in **training** and the testing data in **testing**. Both are enumerations of **Digit** instances.

But CNTK can't train on an enumeration of class instances. It requires **float[][]** values for both features and labels.

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

Note that **training_labels** and **testing_labels** are both **float[][]** values, whereas in previous assignments they were **float[]** values. This is because we're now doing **multiclass** classification. Each output label is no longer a single value but an array with 10 probabilities for each possible digit value.

Now we need to tell CNTK what shape the input data has that we'll train the neural network on, and what shape the output data of the neural network will have: 

```csharp
// build features and labels
var features = NetUtil.Var(new int[] { 28, 28 }, DataType.Float);
var labels = NetUtil.Var(new int[] { 10 }, DataType.Float);

// the rest of the code goes here...
```

Note the first **Var** method which tells CNTK that our neural network will use a 2-dimensional tensor of 28 by 28 floating point pixel values as input. This shape matches the 784 values returned by the **Digit.GetFeatures** method. 

Note that this shape refers to a 2-dimensional tensor of 28x28 values while the **GetFeatures** method returns a 1-dimensional array of 784 values. This is not a problem. The CNTK library will automatically reshape the 1-dimensional array to a 2-dimensional tensor for us.  

The second **Var** method tells CNTK that we want our neural network to output a 1-dimensional tensor of 10 float values. This shape matches the 10 values returned by the **Digit.GetLabel** method.

Our next step is to design the neural network. 

We will use a deep neural network with a 512-node input layer and a 10-node output layer. We'll use the **ReLU** activation function for the input layer and **Softmax** activation for the output layer. 

Remember: the sofmax function creates a mutually exclusive list of output classes where only a single class can be the correct answer. If we had used sigmoid, the neural network might predict more than one digit value simultaneously. We don't want that here.  

Here's how to build the neural network:

```csharp
// build the network
var network = features
    .Dense(512, CNTKLib.ReLU)
    .Dense(10, CNTKLib.Softmax)
    .ToNetwork();
Console.WriteLine("Model architecture:");
Console.WriteLine(network.ToSummary());

// the rest of the code goes here...
```

Each **Dense** call adds a new dense feedforward layer to the network. We're stacking one layer with **ReLU** activation and one layer with **Softmax** activation.

Then we use the **ToSummary** method to output a description of the architecture of the neural network to the console.

Now we need to decide which loss function to use to train the neural network, and how we are going to track the prediction error of the network during each training epoch. 

For this assignment we'll use **CrossEntropyWithSoftmax** as the loss function because it's the standard metric for measuring multiclass classification loss with softmax. 

We'll track the error with the **ClassificationError** metric. This is the number of times (expressed as a percentage) that the model predictions are wrong. An error of 0 means the predictions are correct all the time, and an error of 1 means the predictions are wrong all the time. 

Note that in the previous assignment we used **BinaryClassificationError**. We can't use that function here because we're doing multiclass classification now.

```csharp
// set up the loss function and the classification error function
var lossFunc = CNTKLib.CrossEntropyWithSoftmax(network.Output, labels);
var errorFunc = CNTKLib.ClassificationError(network.Output, labels);

// the rest of the code goes here...
```

Next we need to decide which algorithm to use to train the neural network. There are many possible algorithms derived from Gradient Descent that we can use here.

For this assignment we're going to use the **RMSPropLearner**. You can learn more about the RMS algorithm here: [https://towardsdatascience.com/understanding...](https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a)

```csharp
// set up a trainer that uses the RMSProp algorithm
var learner = network.GetRMSPropLearner(
    learningRateSchedule: 0.99,
    gamma: 0.95,
    inc: 2.0,
    dec: 0.5,
    max: 2.0,
    min: 0.5
);

// the rest of the code goes here...
```

These configuration values are a good starting point, but you can tweak them if you like to try and improve the quality of your predictions.

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
var maxEpochs = 50;
var batchSize = 128;
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

We're training the network for 50 epochs using a batch size of 128. During training we'll track the loss and errors in the **loss**, **trainingError** and **testingError** arrays.

Once training is done, we show the final testing error on the console. This is the percentage of mistakes the network makes when predicting digits. 

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

The **Index().Shuffle().Batch()** sequence randomizes the data and splits it up in a collection of 128-record batches. The second argument to **Batch()** is a function that will be called for every batch.

Inside the batch function we call **GetBatch** twice to get a feature batch and a corresponding label batch. Then we call **TrainBatch** to train the neural network on these two batches of training data.

The **TrainBatch** method returns the loss and error, but only for training on the 128-record batch. So we simply add up all these values and divide them by the number of batches in the dataset. That gives us the average loss and error for the predictions on the training partition during the current epoch, and we report this to the console.

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

We call **TestBatch** to test the neural network on the 128-record test batch. The method returns the error for the batch, and we again add up the errors for each batch and divide by the number of batches. 

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
chart.WithTitle("Digit Training");

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

Now go to the Mnist folder and type:

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

Do you think this model is good at predicting handwritten digits?

Try to improve the neural network by changing the network architecture. You can add more nodes or extra layers. You can also changing the number of epochs, the batch size, or the learner parameters. 

Did the changes help? What is the best accuracy you can achieve?

Post your results in our support group.
