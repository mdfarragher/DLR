using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CNTKUtil
{
    /// <summary>
    /// The Gan class contains helper methods for creating generative adversarial networks. 
    /// </summary>
    public static class Gan
    {
        /// <summary>
        /// Create a Gan by combining a generator and a discriminator.
        /// </summary>
        /// <param name="generator">The generator to use.</param>
        /// <param name="discriminator">The discriminator to use.</param>
        /// <returns>A new Gan network constructed out of the generator and discriminator.</returns>
        public static CNTK.Function CreateGan(
            CNTK.Function generator,
            CNTK.Function discriminator)
        {
            return discriminator.Clone(
                CNTK.ParameterCloningMethod.Share,
                replacements: new Dictionary<CNTK.Variable, CNTK.Variable>() { { discriminator.Arguments[0], generator } });
        }

        /// <summary>
        /// Use the generator to create a list of fake images/
        /// </summary>
        /// <param name="generator">The generator to use.</param>
        /// <param name="batchSize">The batch size.</param>
        /// <param name="latentDimensions">The number of dimensions in the latent input vector.</param>
        /// <returns>A list of images created by the generator.</returns>
        public static IList<IList<float>> GenerateImages(
            CNTK.Function generator,
            int batchSize,
            int latentDimensions)
        {
            // set up a Gaussian random number generator
            var random = new Random();
            var gaussianRandom = new GaussianRandom(random);

            // set up randomized input for the generator
            var random_latent_vectors = gaussianRandom.getFloatSamples(batchSize * latentDimensions);
            var random_latent_vectors_nd = new CNTK.NDArrayView(new int[] { latentDimensions, 1, batchSize }, random_latent_vectors, NetUtil.CurrentDevice);
            var generator_inputs = new Dictionary<CNTK.Variable, CNTK.Value>() { { generator.Arguments[0], new CNTK.Value(random_latent_vectors_nd) } };
            var generator_outputs = new Dictionary<CNTK.Variable, CNTK.Value>() { { generator.Output, null } };

            // run the generator and collect the images
            generator.Evaluate(generator_inputs, generator_outputs, NetUtil.CurrentDevice);
            return generator_outputs[generator.Output].GetDenseData<float>(generator.Output);
        }

        /// <summary>
        /// Get a new batch of images to train the discriminator. 
        /// The batch will contain generated images with label 1 and actual images with label 0.
        /// </summary>
        /// <param name="discriminatorVar">The input variable for the discriminator.</param>
        /// <param name="generatedImages">The list of generated images.</param>
        /// <param name="actualImages">The list of actual images.</param>
        /// <param name="batchSize">The batch size.</param>
        /// <param name="start">The start position in the training partition.</param>
        /// <returns>A tuple with the feature batch and label batch for training.</returns>
        public static (CNTK.Value featureBatch, CNTK.Value labelBatch) GetTrainingBatch(
            CNTK.Variable discriminatorVar,
            IList<IList<float>> generatedImages,
            float[][] actualImages,
            int batchSize,
            int start)
        {
            // set up a Gaussian random number generator
            var random = new Random();
            var gaussianRandom = new GaussianRandom(random);

            // create a training batch for the discriminator
            // the first half of the mini-batch are the fake images (marked with label='1')
            // the second half are real images (marked with label='0')
            var combined_images = new float[2 * batchSize][];
            var labels = new float[2 * batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                combined_images[i] = generatedImages[i].ToArray();
                labels[i] = (float)(1 + 0.05 * gaussianRandom.NextGaussian());

                combined_images[i + batchSize] = actualImages[start + i];
                labels[i + batchSize] = (float)(0.05 * gaussianRandom.NextGaussian());
            }

            // create batches
            var combined_images_minibatch = discriminatorVar.GetBatch(combined_images, 0, combined_images.Length);
            var labels_minibatch = CNTK.Value.CreateBatch(new CNTK.NDShape(0), labels, NetUtil.CurrentDevice, true);

            // return results
            return (combined_images_minibatch, labels_minibatch);
        }

        public static (CNTK.Value featureBatch, CNTK.Value labelBatch) GetMisleadingBatch(
            CNTK.Function gan,
            int batchSize,
            int latentDimensions)
        {
            // set up a Gaussian random number generator
            var random = new Random();
            var gaussianRandom = new GaussianRandom(random);

            // prepare a batch to fool the discriminator: we generate fake images
            // but we label them as real with label=0 
            var random_latent_vectors = gaussianRandom.getFloatSamples(batchSize * latentDimensions);
            var misleading_targets = new float[batchSize];
            var random_latent_vectors_nd = new CNTK.NDArrayView(new int[] { latentDimensions, 1, batchSize }, random_latent_vectors, NetUtil.CurrentDevice);

            // return results
            return (
                new CNTK.Value(random_latent_vectors_nd), 
                CNTK.Value.CreateBatch(new CNTK.NDShape(0), misleading_targets, NetUtil.CurrentDevice, true)
            );
        }

        /// <summary>
        /// Save a gan image to disk.
        /// </summary>
        /// <param name="image">The image data to save.</param>
        /// <param name="width">The width of the image.</param>
        /// <param name="height">The height of the image.</param>
        /// <param name="path">The output path to write to.</param>
        public static void SaveImage(float[] image, int width, int height, string path)
        {
            var img_bytes = StyleTransfer.UnflattenByChannel(image, scaling: 255, invertOrder: true);
            using (var mat = new OpenCvSharp.Mat(height, width, OpenCvSharp.MatType.CV_8UC3, img_bytes, 3 * width))
            {
                mat.SaveImage(path);
            }
        }


    }
}
