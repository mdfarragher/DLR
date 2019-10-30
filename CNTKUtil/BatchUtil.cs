using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CNTKUtil
{
    /// <summary>
    /// A collection of utilities for working with batches. 
    /// </summary>
    public static class BatchUtil
    {
        /// <summary>
        /// Create an index for the given data array.
        /// </summary>
        /// <param name="data">The data array to use.</param>
        /// <returns>An array instance with the numbers 0..N, with N the size of the data array.</returns>
        public static int[] Index(this float[][] data)
        {
            var array = new int[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                array[i] = i;
            }
            return array;
        }

        /// <summary>
        /// Swap two elements in an array.
        /// </summary>
        /// <typeparam name="T">The type of array elements</typeparam>
        /// <param name="array">The array to swap elements in</param>
        /// <param name="n">The index of the first element to swap</param>
        /// <param name="k">The index of the second element to swap</param>
        private static void Swap<T>(T[] array, int n, int k)
        {
            var temp = array[n];
            array[n] = array[k];
            array[k] = temp;
        }

        /// <summary>
        /// Shuffle the given array.
        /// </summary>
        /// <param name="array">The array to shuffle.</param>
        /// <returns>The shuffled array.</returns>
        public static int[] Shuffle(this int[] array)
        {
            var rng = new Random();
            var n = array.Length;
            while (n > 1)
            {
                var k = rng.Next(n--);
                Swap(array, n, k);
            }
            return array;
        }

        /// <summary>
        /// Partition the indices into a set of batches and call an action on each batch.
        /// </summary>
        /// <param name="indices">The indices to use.</param>
        /// <param name="batchSize">The size of each batch.</param>
        /// <param name="action">The action to perform on each batch.</param>
        public static void Batch(
            this int[] indices,
            int batchSize,
            Action<int[], int, int> action)
        {
            var begin = 0;
            while (begin < indices.Length)
            {
                var end = Math.Min(begin + batchSize, indices.Length);
                action(indices, begin, end);
                begin = end;

            }
        }

        /// <summary>
        /// Partition the data array into a set of batches and call an action on each batch.
        /// </summary>
        /// <param name="data">The data array to use.</param>
        /// <param name="batchSize">The size of each batch.</param>
        /// <param name="action">The action to perform on each batch.</param>
        public static void Batch(
            this float[][] data,
            int batchSize,
            Action<float[][], int, int> action)
        {
            var begin = 0;
            while (begin < data.Length)
            {
                var end = Math.Min(begin + batchSize, data.Length);
                action(data, begin, end);
                begin = end;

            }
        }

        /// <summary>
        /// Partition the indices into a set of KFold partitions and call an action on each partition.
        /// </summary>
        /// <param name="indices">The indices to use.</param>
        /// <param name="numFolds">The number of KFold partitions to create.</param>
        /// <param name="action">The action to perform on each partition.</param>
        public static void KFold(
            this int[] indices,
            int numFolds,
            Action<int, int[], int[]> action)
        {
            var foldSize = indices.Length / numFolds;
            for (int i = 0; i < numFolds; i++)
            {
                var validationIndices = Enumerable.Range(i * foldSize, foldSize).ToArray();
                var trainingIndices1 = Enumerable.Range(0, i * foldSize);
                var trainingIndices2 = Enumerable.Range((i + 1) * foldSize, indices.Length - (i + 1) * foldSize);
                var trainingIndices = trainingIndices1.Concat(trainingIndices2).ToArray();
                action(i, trainingIndices, validationIndices);
            }
        }

        /// <summary>
        /// Get a batch from the given variable.
        /// </summary>
        /// <param name="variable">The variable to use.</param>
        /// <param name="source">The variable data.</param>
        /// <param name="indices">The array of data indices to use.</param>
        /// <param name="begin">The first index to use.</param>
        /// <param name="end">The last index to use.</param>
        /// <returns>A batch of values taken from the given variable.</returns>
        public static CNTK.Value GetBatch(
            this CNTK.Variable variable,
            float[][] source,
            int[] indices,
            int begin,
            int end)
        {
            var num_indices = end - begin;
            var row_length = variable.Shape.TotalSize;
            var result = new CNTK.NDArrayView[num_indices];

            var row_index = 0;
            for (var index = begin; index != end; index++)
            {
                var dataBuffer = source[indices[index]];
                var ndArrayView = new CNTK.NDArrayView(variable.Shape, dataBuffer, CNTK.DeviceDescriptor.CPUDevice, true);
                result[row_index++] = ndArrayView;
            }
            return CNTK.Value.Create(variable.Shape, result, NetUtil.CurrentDevice, true);
        }

        /// <summary>
        /// Get a batch from the given variable.
        /// </summary>
        /// <param name="variable">The variable to use.</param>
        /// <param name="source">The variable data.</param>
        /// <param name="indices">The array of data indices to use.</param>
        /// <param name="begin">The first index to use.</param>
        /// <param name="end">The last index to use.</param>
        /// <returns>A batch of values taken from the given variable.</returns>
        public static CNTK.Value GetBatch(
            this CNTK.Variable variable,
            float[] source,
            int[] indices,
            int begin,
            int end)
        {
            var num_indices = end - begin;
            var row_length = variable.Shape.TotalSize;
            var result = new float[num_indices];
            var row_index = 0;
            for (var index = begin; index != end; index++)
            {
                result[row_index++] = source[indices[index]];
            }
            return CNTK.Value.CreateBatch(variable.Shape, result, NetUtil.CurrentDevice, true);
        }

        /// <summary>
        /// Get a batch from the given variable.
        /// </summary>
        /// <param name="variable">The variable to use.</param>
        /// <param name="source">The variable data.</param>
        /// <param name="begin">The index of the first value to use.</param>
        /// <param name="end">The index of the last value to use.</param>
        /// <returns>A batch of values taken from the given variable.</returns>
        public static CNTK.Value GetBatch(
            this CNTK.Variable variable,
            float[][] source,
            int begin,
            int end)
        {
            var num_indices = end - begin;
            var result = new CNTK.NDArrayView[num_indices];
            var row_index = 0;
            for (var index = begin; index != end; index++)
            {
                var dataBuffer = source[index];
                var ndArrayView = new CNTK.NDArrayView(variable.Shape, dataBuffer, CNTK.DeviceDescriptor.CPUDevice, true);
                result[row_index++] = ndArrayView;
            }
            return CNTK.Value.Create(variable.Shape, result, NetUtil.CurrentDevice, true);
        }

        /// <summary>
        /// Get a batch from the given variable.
        /// </summary>
        /// <param name="variable">The variable to use.</param>
        /// <param name="source">The variable data.</param>
        /// <param name="begin">The index of the first value to use.</param>
        /// <param name="end">The index of the last value to use.</param>
        /// <returns>A batch of values taken from the given variable.</returns>
        public static CNTK.Value GetBatch(
            this CNTK.Variable variable,
            float[] source,
            int begin,
            int end)
        {
            var result = new float[end - begin];
            Array.Copy(source, begin, result, 0, result.Length);
            return CNTK.Value.CreateBatch(variable.Shape, result, NetUtil.CurrentDevice, true);
        }

        /// <summary>
        /// Get a sequence batch from the given variable.
        /// </summary>
        /// <param name="variable">The variable to use.</param>
        /// <param name="sequenceLength">The number of time periods in the data sequence.</param>
        /// <param name="source">The variable data.</param>
        /// <param name="begin">The index of the first value to use.</param>
        /// <param name="end">The index of the last value to use.</param>
        /// <returns>A batch of values taken from the given variable.</returns>
        public static CNTK.Value GetSequenceBatch(
            this CNTK.Variable variable,
            int sequenceLength,
            float[][] source,
            int begin,
            int end)
        {
            System.Diagnostics.Debug.Assert((variable.Shape.Dimensions.Count == 0) || ((variable.Shape.Dimensions.Count == 1) && (variable.Shape.Dimensions[0] == 1)));
            System.Diagnostics.Debug.Assert(source[0].Length == sequenceLength);
            var num_indices = end - begin;
            var cpu_blob = new float[num_indices * sequenceLength];
            var row_index = 0;
            for (var index = begin; index != end; index++)
            {
                System.Buffer.BlockCopy(source[index], 0, cpu_blob, row_index * sequenceLength * sizeof(float), sequenceLength * sizeof(float));
                row_index++;
            }
            var blob_shape = variable.Shape.AppendShape(new int[] { sequenceLength, end - begin });
            var ndArrayView = new CNTK.NDArrayView(blob_shape, cpu_blob, NetUtil.CurrentDevice);
            return new CNTK.Value(ndArrayView);
        }

        /// <summary>
        /// Get a sequence batch from the given variable.
        /// </summary>
        /// <param name="variable">The variable to use.</param>
        /// <param name="sequenceLength">The number of time periods in the data sequence.</param>
        /// <param name="source">The variable data.</param>
        /// <param name="indices">The array of data indices to use.</param>
        /// <param name="begin">The first index to use.</param>
        /// <param name="end">The Last index to use.</param>
        /// <returns>A batch of values taken from the given variable.</returns>
        public static CNTK.Value GetSequenceBatch(
            this CNTK.Variable variable,
            int sequenceLength,
            float[][] source,
            int[] indices,
            int begin,
            int end)
        {
            System.Diagnostics.Debug.Assert((variable.Shape.Dimensions.Count == 0) || ((variable.Shape.Dimensions.Count == 1) && (variable.Shape.Dimensions[0] == 1)));
            System.Diagnostics.Debug.Assert(source[0].Length == sequenceLength);
            var num_indices = end - begin;
            var cpu_blob = new float[num_indices * sequenceLength];
            var row_index = 0;
            for (var index = begin; index != end; index++)
            {
                System.Buffer.BlockCopy(source[indices[index]], 0, cpu_blob, row_index * sequenceLength * sizeof(float), sequenceLength * sizeof(float));
                row_index++;
            }
            var blob_shape = variable.Shape.AppendShape(new int[] { sequenceLength, end - begin });
            var ndArrayView = new CNTK.NDArrayView(blob_shape, cpu_blob, NetUtil.CurrentDevice);
            return new CNTK.Value(ndArrayView);
        }

        /// <summary>
        /// Get a sequence batch from the given variable.
        /// </summary>
        /// <param name="variable">The variable to use.</param>
        /// <param name="sequenceLength">The number of time periods in the data sequence.</param>
        /// <param name="source">The variable data.</param>
        /// <param name="begin">The index of the first value to use.</param>
        /// <param name="end">The index of the last value to use.</param>
        /// <returns>A batch of values taken from the given variable.</returns>
        public static CNTK.Value GetSequenceBatch(
            this CNTK.Variable variable,
            int sequenceLength,
            float[] source, 
            int begin, 
            int end)
        {
            System.Diagnostics.Debug.Assert((variable.Shape.Dimensions.Count == 0) || ((variable.Shape.Dimensions.Count == 1) && (variable.Shape.Dimensions[0] == 1)));
            var num_indices = end - begin;
            var cpu_tensors = new float[num_indices][];
            var row_index = 0;
            for (var index = begin; index != end; index++)
            {
                cpu_tensors[row_index] = new float[sequenceLength];
                cpu_tensors[row_index][sequenceLength - 1] = source[index];
                row_index++;
            }
            var result = CNTK.Value.CreateBatchOfSequences(variable.Shape, cpu_tensors, NetUtil.CurrentDevice, true);
            return result;
        }

        /// <summary>
        /// Get a batch from the given image reader.
        /// </summary>
        /// <param name="reader">The image reader to use.</param>
        /// <param name="batchSize">The size of the batch.</param>
        /// <returns>A batch of values taken from the given image reader.</returns>
        public static CNTK.UnorderedMapStreamInformationMinibatchData GetBatch(
            this CNTK.MinibatchSource reader,
            int batchSize)
        {
            return reader.GetNextMinibatch((uint)batchSize, NetUtil.CurrentDevice);
        }
    }
}
