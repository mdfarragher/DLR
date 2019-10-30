using System;
using System.Collections.Generic;
using System.Text;

namespace CNTKUtil
{
    /// <summary>
    /// The GaussianRandom class generates random numbers along a Gaussian distribution.
    /// </summary>
    public sealed class GaussianRandom
    {
        private bool _hasDeviate;
        private double _storedDeviate;
        private readonly Random _random;

        public GaussianRandom(Random random = null)
        {
            _random = random ?? new Random();
        }

        public float[] getFloatSamples(int numSamples)
        {
            var result = new float[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                result[i] = (float)NextGaussian();
            }
            return result;
        }

        public double NextGaussian(double mu = 0, double sigma = 1)
        {
            if (sigma <= 0)
                throw new ArgumentOutOfRangeException("sigma", "Must be greater than zero.");

            if (_hasDeviate)
            {
                _hasDeviate = false;
                return _storedDeviate * sigma + mu;
            }

            double v1, v2, rSquared;
            do
            {
                v1 = 2 * _random.NextDouble() - 1;
                v2 = 2 * _random.NextDouble() - 1;
                rSquared = v1 * v1 + v2 * v2;
            } while (rSquared >= 1 || rSquared == 0);

            var polar = Math.Sqrt(-2 * Math.Log(rSquared) / rSquared);
            _storedDeviate = v2 * polar;
            _hasDeviate = true;
            return v1 * polar * sigma + mu;
        }
    }
}
