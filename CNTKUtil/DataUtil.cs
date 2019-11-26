using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace CNTKUtil
{
    /// <summary>
    /// A collection of utilities to work with data files.
    /// </summary>
    public static class DataUtil
    {
        /// <summary>
        /// Load the given binary file from disk.
        /// </summary>
        /// <param name="filepath">The filename of the file to load.</param>
        /// <param name="numRows">The number of rows to load.</param>
        /// <param name="numColumns">The number of columns to load.</param>
        /// <returns></returns>
        public static T[][] LoadBinary<T>(
            string filepath,
            int numRows,
            int numColumns)
        {
            var size = Marshal.SizeOf(typeof(T)); // warning: unreliable for char!
            var buffer = new byte[size * numRows * numColumns];
            using (var reader = new System.IO.BinaryReader(System.IO.File.OpenRead(filepath)))
            {
                reader.Read(buffer, 0, buffer.Length);
            }
            var dst = new T[numRows][];
            for (int row = 0; row < dst.Length; row++)
            {
                dst[row] = new T[numColumns];
                Buffer.BlockCopy(buffer, row * numColumns * size, dst[row], 0, numColumns * size);
            }
            return dst;
        }

        /// <summary>
        /// Load the given binary file from disk.
        /// </summary>
        /// <param name="filepath">The filename of the file to load.</param>
        /// <param name="numRows">The number of rows to load.</param>
        /// <returns></returns>
        public static T[] LoadBinary<T>(
            string filepath,
            int numRows)
        {
            var size = Marshal.SizeOf(typeof(T));
            var buffer = new byte[size * numRows];
            using (var reader = new System.IO.BinaryReader(System.IO.File.OpenRead(filepath)))
            {
                reader.Read(buffer, 0, buffer.Length);
            }
            var dst = new T[numRows];
            System.Buffer.BlockCopy(buffer, 0, dst, 0, buffer.Length);
            return dst;
        }

        /// <summary>
        /// The pretrained VGG16 image classifier.
        /// </summary>
        public static class VGG16
        {
            // private members
            private const string Filename = "VGG16_ImageNet_Caffe.model";
            private const string DownloadUrl = "https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet_Caffe.model";

            /// <summary>
            /// The full path to the downloaded model.
            /// </summary>
            private static string GetFullPath()
            {
                return Path.Combine(Directory.GetCurrentDirectory(), Path.Combine("models", Filename));
            }

            /// <summary>
            /// Get if the model is already downloaded from the internet.
            /// </summary>
            public static bool IsDownloaded
            {
                get {
                    string fullPath = GetFullPath();
                    return File.Exists(fullPath);
                }
            }

            /// <summary>
            /// Download the model from the internet.
            /// </summary>
            public static void Download()
            {
                string fullPath = GetFullPath();
                DataUtil.DownloadModel(fullPath, DownloadUrl);
            }

            /// <summary>
            /// Load the model from disk.
            /// </summary>
            /// <param name="features">The input features for the model.</param>
            /// <param name="allowBlock5Finetuning">Set to true to allow finetuning of convolution block 5.</param>
            /// <returns>The fully trained VGG16 model.</returns>
            public static CNTK.Function GetModel(CNTK.Variable features, bool allowBlock5Finetuning = false)
            {
                // make sure the model has been downloaded
                if (!IsDownloaded)
                    Download();

                // load the model into a new function
                string fullPath = GetFullPath();
                var model = CNTK.Function.Load(fullPath, NetUtil.CurrentDevice);

                // get the last VGG16 layer before the first fully connected layer
                var last_frozen_layer = model.FindByName(allowBlock5Finetuning ? "pool4" : "pool5");

                // get the first layer, and the "data" input variable
                var conv1_1_layer = model.FindByName("conv1_1");
                var data = conv1_1_layer.Inputs.First((v) => v.Name == "data");

                // the data should be a 224x224x3 input tensor
                if (!data.Shape.Dimensions.SequenceEqual(new int[] { 224, 224, 3 }))
                {
                    throw new InvalidOperationException("There's a problem here. Please email");
                }

                // allow different dimensions for input (e.g., 150x150x3)
                var replacements = new Dictionary<CNTK.Variable, CNTK.Variable>() { { data, features } };

                // clone the original VGG16 model up to the pool_node, freeze all weights, and use a custom input tensor
                var frozen_model = CNTK.CNTKLib
                  .Combine(new CNTK.VariableVector() { last_frozen_layer.Output }, "frozen_output")
                  .Clone(CNTK.ParameterCloningMethod.Freeze, replacements);

                // stop here if we're not finetuning
                if (!allowBlock5Finetuning)
                {
                    return frozen_model;
                }

                // enable finetuning for block 5
                var pool5_layer = model.FindByName("pool5");
                replacements = new Dictionary<CNTK.Variable, CNTK.Variable>() { { last_frozen_layer.Output, frozen_model.Output } };

                var model_with_finetuning = CNTK.CNTKLib
                  .Combine(new CNTK.VariableVector() { pool5_layer.Output }, "finetuning_output")
                  .Clone(CNTK.ParameterCloningMethod.Clone, replacements);

                // return model with finetuning
                return model_with_finetuning;
            }
        }

        /// <summary>
        /// The pretrained VGG19 image classifier.
        /// </summary>
        public static class VGG19
        {
            // private members
            private const string Filename = "VGG19_ImageNet_Caffe.model";
            private const string DownloadUrl = "https://www.cntk.ai/Models/Caffe_Converted/VGG19_ImageNet_Caffe.model";

            /// <summary>
            /// The full path to the downloaded model.
            /// </summary>
            private static string GetFullPath()
            {
                return Path.Combine(Directory.GetCurrentDirectory(), Path.Combine("models", Filename));
            }

            /// <summary>
            /// Get if the model is already downloaded from the internet.
            /// </summary>
            public static bool IsDownloaded
            {
                get
                {
                    string fullPath = GetFullPath();
                    return File.Exists(fullPath);
                }
            }

            /// <summary>
            /// Download the model from the internet.
            /// </summary>
            public static void Download()
            {
                string fullPath = GetFullPath();
                DataUtil.DownloadModel(fullPath, DownloadUrl);
            }

            /// <summary>
            /// Load the model from disk.
            /// </summary>
            /// <param name="features">The input features for the model.</param>
            /// <param name="freeze">Set to true to freeze all weights in the network.</param>
            /// <returns>The fully trained VGG16 model.</returns>
            public static CNTK.Function GetModel(CNTK.Variable features, bool freeze = false)
            {
                // make sure the model has been downloaded
                if (!IsDownloaded)
                    Download();

                // load the model into a new function
                string fullPath = GetFullPath();
                var model = CNTK.Function.Load(fullPath, NetUtil.CurrentDevice);

                // return the model up to the 'pool5' layer, without feature replacements
                var cloningMethod = freeze ? CNTK.ParameterCloningMethod.Freeze : CNTK.ParameterCloningMethod.Clone;
                var pool5_node = model.FindByName("pool5");
                CNTK.Function cloned_model = null;
                if (features == null)
                {
                    cloned_model = CNTK.Function.Combine(new CNTK.Variable[] { pool5_node }).Clone(cloningMethod);
                    return cloned_model;
                }

                // return the model up to the 'pool5' layer, with feature replacements
                System.Diagnostics.Debug.Assert(model.Arguments.Count == 1);
                var replacements = new Dictionary<CNTK.Variable, CNTK.Variable>() { { model.Arguments[0], features } };
                cloned_model = CNTK.Function.Combine(new CNTK.Variable[] { pool5_node }).Clone(cloningMethod, replacements);
                return cloned_model;
            }
        }

        /// <summary>
        /// Get an image reader to sequentially read images from disk for training.
        /// </summary>
        /// <param name="mapFilePath">The path to the map file</param>
        /// <param name="imageWidth">The width to scale all images to</param>
        /// <param name="imageHeight">The height to scale all images to</param>
        /// <param name="numChannels">The number of channels to transform all images to</param>
        /// <param name="numClasses">The number of label classes in this training set</param>
        /// <param name="randomizeData">Set to true to randomize the data for training</param>
        /// <param name="augmentData">Set to true to use data augmentation to expand the training set</param>
        /// <returns>An image source ready for use in training or testing.</returns>
        public static CNTK.MinibatchSource GetImageReader(string mapFilePath, int imageWidth, int imageHeight, int numChannels, int numClasses, bool randomizeData, bool augmentData)
        {
            var transforms = new List<CNTK.CNTKDictionary>();
            if (augmentData)
            {
                var randomSideTransform = CNTK.CNTKLib.ReaderCrop("RandomSide",
                  new Tuple<int, int>(0, 0),
                  new Tuple<float, float>(0.8f, 1.0f),
                  new Tuple<float, float>(0.0f, 0.0f),
                  new Tuple<float, float>(1.0f, 1.0f),
                  "uniRatio");
                transforms.Add(randomSideTransform);
            }
            var scaleTransform = CNTK.CNTKLib.ReaderScale(imageWidth, imageHeight, numChannels);
            transforms.Add(scaleTransform);

            var imageDeserializer = CNTK.CNTKLib.ImageDeserializer(mapFilePath, "labels", (uint)numClasses, "features", transforms);
            var minibatchSourceConfig = new CNTK.MinibatchSourceConfig(new CNTK.DictionaryVector() { imageDeserializer });
            if (!randomizeData)
            {
                minibatchSourceConfig.randomizationWindowInChunks = 0;
                minibatchSourceConfig.randomizationWindowInSamples = 0;
            }
            return CNTK.CNTKLib.CreateCompositeMinibatchSource(minibatchSourceConfig);
        }

        /// <summary>
        /// Download a pretrained model from the internet.
        /// </summary>
        /// <param name="fullPath">The path where to store the model locally.</param>
        /// <param name="downloadUrl">The url where to download the model from.</param>
        /// <returns>Returns true if the download was successful, or false if it was not.</returns>
        public static bool DownloadModel(string fullPath, string downloadUrl)
        {
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            if (File.Exists(fullPath))
            {
                return true; // file already exists
            }
            var success = FileDownloader.DownloadFile(downloadUrl, fullPath, timeoutInMilliSec: 3600000);
            return success;
        }

    }
    

    /// <summary>
    /// A downloader class to download files from the internet. 
    /// </summary>
    public class FileDownloader
    {
        private readonly string _url;
        private readonly string _fullPathWhereToSave;
        private bool _result = false;
        private readonly SemaphoreSlim _semaphore = new SemaphoreSlim(0);

        public FileDownloader(string url, string fullPathWhereToSave)
        {
            if (string.IsNullOrEmpty(url)) throw new ArgumentNullException("url");
            if (string.IsNullOrEmpty(fullPathWhereToSave)) throw new ArgumentNullException("fullPathWhereToSave");

            this._url = url;
            this._fullPathWhereToSave = fullPathWhereToSave;
        }

        public bool StartDownload(int timeout)
        {
            try
            {
                System.IO.Directory.CreateDirectory(Path.GetDirectoryName(_fullPathWhereToSave));

                if (File.Exists(_fullPathWhereToSave))
                {
                    File.Delete(_fullPathWhereToSave);
                }
                using (WebClient client = new WebClient())
                {
                    var ur = new Uri(_url);
                    // client.Credentials = new NetworkCredential("username", "password");
                    client.DownloadProgressChanged += WebClientDownloadProgressChanged;
                    client.DownloadFileCompleted += WebClientDownloadCompleted;
                    Console.WriteLine("    Downloading " + ur);
                    client.DownloadFileAsync(ur, _fullPathWhereToSave);
                    _semaphore.Wait(timeout);
                    return _result && File.Exists(_fullPathWhereToSave);
                }
            }
            catch (Exception e)
            {
                throw e;
            }
            finally
            {
                this._semaphore.Dispose();
            }
        }

        private void WebClientDownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            Console.Write("\r     -->    {0}%.", e.ProgressPercentage);
        }

        private void WebClientDownloadCompleted(object sender, AsyncCompletedEventArgs args)
        {
            _result = !args.Cancelled;
            if (!_result)
            {
                throw new IOException(args.Error.ToString());
            }
            _semaphore.Release();
        }

        public static bool DownloadFile(string url, string fullPathWhereToSave, int timeoutInMilliSec)
        {
            return new FileDownloader(url, fullPathWhereToSave).StartDownload(timeoutInMilliSec);
        }
    }
}
