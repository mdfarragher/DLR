using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CNTKUtil
{
    /// <summary>
    /// The StyleTransfer class contains methods for artistic style transfer.
    /// </summary>
    public static class StyleTransfer
    {
        // color channel offsets for the VGG19 network
        static readonly float[] VGG19_Offsets = new float[] { 103.939f, 116.779f, 123.68f };

        /// <summary>
        /// Convert an image to a 1-dimensional float array. All color frames
        /// are laid out sequentially, one after the other. 
        /// </summary>
        /// <param name="mat">The input image as an OpenCv Mat object.</param>
        /// <param name="offsets">Offsets to apply to each color channel.</param>
        /// <returns>An 1-dimensional float array containing the image data.</returns>
        public static float[] FlattenByChannel(Mat mat, float[] offsets)
        {
            var num_pixels = mat.Size().Height * mat.Size().Width;
            float[] result = new float[num_pixels * 3];
            //using (MatOfByte3 mat3 = new MatOfByte3(mat))
            using (Mat<Vec3b> mat3 = new Mat<Vec3b>(mat))
            {
                var indexer = mat3.GetIndexer();
                var pos = 0;
                for (int y = 0; y < mat.Height; y++)
                {
                    for (int x = 0; x < mat.Width; x++)
                    {
                        var color = indexer[y, x];
                        result[pos] = color.Item0 - offsets[0];
                        result[pos + num_pixels] = color.Item1 - offsets[1];
                        result[pos + 2 * num_pixels] = color.Item2 - offsets[2];
                        pos++;
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Convert an 1-dimensional float array to an image.
        /// </summary>
        /// <param name="offsets,g">The float array to process.</param>
        /// <param name="offsets">Offsets to apply to each color channel.</param>
        /// <param name="scaling">The scaling factor to apply.</param>
        /// <param name="invertOrder">Set to invert the order of the channels.</param>
        /// <returns>An image converted from the 1-dimensional float array .</returns>
        public static byte[] UnflattenByChannel(float[] img, float[] offsets = null, float scaling = 1.0f, bool invertOrder = false)
        {
            if (offsets == null) { offsets = new float[3]; }
            var img_data = new byte[img.Length];
            var image_size = img.Length / 3;
            for (int i = 0; i < img_data.Length; i += 3)
            {
                img_data[i + 1] = (byte)Math.Max(0, Math.Min(scaling * img[i / 3 + image_size] + offsets[1], 255));
                if (invertOrder)
                {
                    img_data[i + 2] = (byte)Math.Max(0, Math.Min(scaling * img[i / 3] + offsets[0], 255));
                    img_data[i] = (byte)Math.Max(0, Math.Min(scaling * img[i / 3 + 2 * image_size] + offsets[2], 255));
                }
                else
                {
                    img_data[i] = (byte)Math.Max(0, Math.Min(scaling * img[i / 3] + offsets[0], 255));
                    img_data[i + 2] = (byte)Math.Max(0, Math.Min(scaling * img[i / 3 + 2 * image_size] + offsets[2], 255));
                }
            }
            return img_data;
        }

        /// <summary>
        /// Load an image from disk and return it as a 1-dimensional float array
        /// with all color channels laid out sequentially.
        /// </summary>
        /// <param name="imagePath">The path of the file to load.</param>
        /// <param name="width">the width of the file.</param>
        /// <param name="height">The height of the file.</param>
        /// <returns></returns>
        public static float[] LoadImage(
            string imagePath,
            int width,
            int height)
        {
            using (var mat = Cv2.ImRead(imagePath))
            {
                using (var mat2 = new Mat(height, width, mat.Type()))
                {
                    Cv2.Resize(mat, mat2, new Size(width, height));
                    return FlattenByChannel(mat2, VGG19_Offsets);
                }
            }
        }

        /// <summary>
        /// Get a list of content and style layers from the given neural network.
        /// </summary>
        /// <param name="input">The neural network to process.</param>
        /// <returns>A list of content and style layers in the neural network.</returns>
        public static List<CNTK.Variable> GetContentAndStyleLayers(
            this CNTK.Function input)
        {
            var layers = new List<CNTK.Variable>();
            var layerNames = new string[] { "conv5_2", "conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1" };
            foreach (var layerName in layerNames)
            {
                var layer = input.FindByName(layerName);
                layers.Add(layer);
            }
            return layers;
        }

        /// <summary>
        /// Create a style transfer base model from a VGG19 network.
        /// </summary>
        /// <param name="feature">The input feature to use.</param>
        /// <returns>A neural network containing the content and style layers of the VGG19 network.</returns>
        public static CNTK.Function StyleTransferBase(
            this CNTK.Variable input)
        {
            // extract all content and style layers
            var layers = ((CNTK.Function)input).GetContentAndStyleLayers();

            // return a model from these layers
            return CNTK.Function.Combine(layers, "content_and_styles").Clone(CNTK.ParameterCloningMethod.Freeze);
        }

        /// <summary>
        /// Adds a dream layer to a neural network.
        /// </summary>
        /// <param name="input">The neural network to extend.</param>
        /// <param name="image">The content image.</param>
        /// <param name="width">The width of the content image.</param>
        /// <param name="height">The height of the content image.</param>
        /// <returns>The neural network extended with a dream layer.</returns>
        public static CNTK.Function DreamLayer(
            this CNTK.Function input,
            float[] image,
            int width,
            int height)
        {
            // set up the dream layer
            var dream_weights_init = new CNTK.NDArrayView(new int[] { width, height, 3 }, image, NetUtil.CurrentDevice);
            var dream_weights = new CNTK.Parameter(dream_weights_init, "the_dream");
            var dummy_features = CNTK.Variable.InputVariable(new int[] { 1 }, CNTK.DataType.Float, "dummy_features");
            var dream_layer = CNTK.CNTKLib.ElementTimes(dream_weights, dummy_features, "the_dream_layer");

            // combine the dream layer with the content and style layers
            var replacements = new Dictionary<CNTK.Variable, CNTK.Variable>() { { input.Arguments[0], dream_layer.Output } };
            var model = input.Clone(CNTK.ParameterCloningMethod.Freeze, replacements);

            // return the finished model
            var all_outputs = new List<CNTK.Variable>() { dream_layer };
            all_outputs.AddRange(model.Outputs);
            return CNTK.Function.Combine(all_outputs, name: "overall_model");
        }

        /// <summary>
        /// Calculate the output labels for style transfer.
        /// </summary>
        /// <param name="model">The neural network to use.</param>
        /// <param name="contentImage">The content image to use.</param>
        /// <param name="styleImage">The style image to use.</param>
        /// <returns></returns>
        public static float[][] CalculateLabels(CNTK.Function model, float[] contentImage, float[] styleImage)
        {
            // make sure the content image dimensions match the neural network input size
            // make sure the content and style images are the same size
            var input_shape = model.Arguments[0].Shape.Dimensions.ToArray();
            System.Diagnostics.Debug.Assert(input_shape[0] * input_shape[1] * input_shape[2] == contentImage.Length);
            System.Diagnostics.Debug.Assert(contentImage.Length == styleImage.Length);

            // set up a batch with the content and the style image
            var batch_buffer = new float[2 * contentImage.Length];
            Array.Copy(contentImage, 0, batch_buffer, 0, contentImage.Length);
            Array.Copy(styleImage, 0, batch_buffer, contentImage.Length, contentImage.Length);
            var batch_nd = new CNTK.NDArrayView(new int[] { model.Arguments[0].Shape[0], model.Arguments[0].Shape[1], model.Arguments[0].Shape[2], 1, 2 }, batch_buffer, NetUtil.CurrentDevice);
            var batch = new CNTK.Value(batch_nd);

            // let the model evaluate the batch
            var inputs = new Dictionary<CNTK.Variable, CNTK.Value>() { { model.Arguments[0], batch } };
            var outputs = new Dictionary<CNTK.Variable, CNTK.Value>();
            foreach (var output in model.Outputs)
            {
                outputs.Add(output, null);
            }
            model.Evaluate(inputs, outputs, NetUtil.CurrentDevice);

            // collect and return the model outputs
            float[][] labels = new float[model.Outputs.Count][];
            labels[0] = outputs[model.Outputs[0]].GetDenseData<float>(model.Outputs[0])[0].ToArray();
            for (int i = 1; i < labels.Length; i++)
            {
                labels[i] = outputs[model.Outputs[i]].GetDenseData<float>(model.Outputs[i])[1].ToArray();
            }
            return labels;
        }

        /// <summary>
        /// Create a loss function that can compare the model output with the style image.
        /// </summary>
        /// <param name="model">The model to use.</param>
        /// <param name="outputs">The model outputs.</param>
        /// <param name="labels">The labels to use.</param>
        /// <returns>The loss function to use for training.</returns>
        public static CNTK.Function CreateLossFunction(
            CNTK.Function model, 
            IList<CNTK.Variable> outputs, 
            IList<CNTK.Variable> labels)
        {
            var lossFunction = ContentLossFunction(outputs[0], labels[0]);
            for (int i = 1; i < outputs.Count; i++)
            {
                var sl = StyleLossFunction(outputs[i], labels[i]);
                lossFunction = CNTK.CNTKLib.Plus(lossFunction, sl);
            }
            return lossFunction;
        }

        /// <summary>
        /// Create a batch for training.
        /// </summary>
        /// <param name="model">The model to use.</param>
        /// <param name="labels">The label data to use.</param>
        /// <returns>A batch for training.</returns>
        public static Dictionary<CNTK.Variable, CNTK.Value> CreateBatch(CNTK.Function model, float[][] labels)
        {
            var dictionary = new Dictionary<CNTK.Variable, CNTK.Value>();
            for (int i = 0; i < model.Arguments.Count; i++)
            {
                var loss_input_variable = model.Arguments[i];
                if (loss_input_variable.Name == "dummy_features")
                {
                    var dummy_scalar_buffer = new float[] { 1 };
                    var dummy_scalar_nd = new CNTK.NDArrayView(new int[] { 1 }, dummy_scalar_buffer, NetUtil.CurrentDevice, readOnly: true);
                    dictionary[loss_input_variable] = new CNTK.Value(dummy_scalar_nd);
                }
                else
                {
                    var cs_index = Int32.Parse(loss_input_variable.Name.Substring("content_and_style_".Length));
                    var nd = new CNTK.NDArrayView(loss_input_variable.Shape, labels[cs_index], NetUtil.CurrentDevice, readOnly: true);
                    dictionary[loss_input_variable] = new CNTK.Value(nd);
                }
            }
            return dictionary;
        }

        /// <summary>
        /// Infer the image from the trained model.
        /// </summary>
        /// <param name="model">The model to use.</param>
        /// <param name="batch">The evaluation batch to use.</param>
        /// <returns>The image inferred from the model.</returns>
        public static byte[] InferImage(
            this CNTK.Function model,
            Dictionary<CNTK.Variable, CNTK.Value> batch)
        {
            var outputs = new Dictionary<CNTK.Variable, CNTK.Value>() { { model.Outputs[0], null } };
            model.Evaluate(batch, outputs, NetUtil.CurrentDevice);
            var img = outputs[model.Outputs[0]].GetDenseData<float>(model.Outputs[0])[0].ToArray();
            return UnflattenByChannel(img, VGG19_Offsets);
        }

        /// <summary>
        /// A content loss function for style transfer.
        /// </summary>
        /// <param name="x">The current content image tensor.</param>
        /// <param name="y">The desired content image tensor.</param>
        /// <returns>A function that can calculate the content loss.</returns>
        private static CNTK.Function ContentLossFunction(CNTK.Variable x, CNTK.Function y)
        {
            var diff_ = CNTK.CNTKLib.Minus(x, y, name: "content_loss_diff_");
            var square_ = CNTK.CNTKLib.Square(diff_, name: "content_loss_square_");
            var sum_ = CNTK.CNTKLib.ReduceSum(square_, CNTK.Axis.AllStaticAxes(), name: "content_loss_sum_");
            var scaling = CNTK.Constant.Scalar((float)(1.0 / x.Shape.TotalSize), NetUtil.CurrentDevice);
            sum_ = CNTK.CNTKLib.ElementTimes(sum_, scaling, name: "content_loss_");
            return sum_;
        }

        /// <summary>
        /// A style loss function for style transfer.
        /// </summary>
        /// <param name="style">The current style image tensor.</param>
        /// <param name="combination">The combination.</param>
        /// <returns>A function that can calculate the style loss.</returns>
        private static CNTK.Function StyleLossFunction(CNTK.Variable style, CNTK.Variable combination)
        {
            var style_gram = GramMatrix(style);
            var combination_gram = GramMatrix(combination);
            var diff_ = CNTK.CNTKLib.Minus(style_gram, combination_gram, name: "style_loss_diff_");
            var square_ = CNTK.CNTKLib.Square(diff_, name: "style_loss_square_");
            var sum_ = CNTK.CNTKLib.ReduceSum(square_, CNTK.Axis.AllStaticAxes(), name: "style_loss_reduce_sum_");
            var max_ = CNTK.CNTKLib.ReduceMax(style_gram, CNTK.Axis.AllStaticAxes(), name: "style_loss_reduce_max");
            var style_gram_total_size = style_gram.Output.Shape.Dimensions[0] * style_gram.Output.Shape.Dimensions[1];
            var scaling_factor = CNTK.Constant.Scalar((float)style_gram_total_size, NetUtil.CurrentDevice);
            var result = CNTK.CNTKLib.ElementDivide(sum_, scaling_factor, name: "style_loss_result_");
            result = CNTK.CNTKLib.ElementDivide(result, max_, name: "style_loss_");
            return result;
        }

        /// <summary>
        /// Return the gramm matrix.
        /// </summary>
        /// <param name="x">The input.</param>
        /// <returns>The gram matrix of the input.</returns>
        private static CNTK.Function GramMatrix(CNTK.Variable x)
        {
            var x_shape = x.Shape.Dimensions.ToArray();
            var features = CNTK.CNTKLib.Reshape(x, new int[] { x_shape[0] * x_shape[1], x_shape[2] }, name: "gram_matrix_reshape_");
            var gram = CNTK.CNTKLib.TransposeTimes(features, features, name: "gram_matrix_transpose_times_");
            return gram;
        }

    }
}
