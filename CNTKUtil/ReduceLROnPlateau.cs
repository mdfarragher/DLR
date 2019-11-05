using System;
using System.Collections.Generic;
using System.Text;

namespace CNTKUtil
{
    /// <summary>
    /// The ReduceLROnPlateau class is a scheduler that dynamically adjusts the 
    /// learning rate when the training curve plateaus.
    /// </summary>
    public class ReduceLROnPlateau
    {
        readonly CNTK.Learner learner;
        double learningRate = 0;
        double bestMetric = 1e-5;
        int slotSinceLastUpdate = 0;

        /// <summary>
        /// Construct a new instance of the class.
        /// </summary>
        /// <param name="learner">The learning algorithm to use.</param>
        /// <param name="lr">The starting learning rate.</param>
        public ReduceLROnPlateau(CNTK.Learner learner, double lr)
        {
            this.learner = learner;
            this.learningRate = lr;
        }

        /// <summary>
        /// Update the learning rate.
        /// </summary>
        /// <param name="current_metric">The current value of the training metric.</param>
        /// <returns>Indicates if training should stop.</returns>
        public bool Update(double current_metric)
        {
            bool should_stop = false;
            if (current_metric < bestMetric)
            {
                bestMetric = current_metric;
                slotSinceLastUpdate = 0;
                return should_stop;
            }
            slotSinceLastUpdate++;
            if (slotSinceLastUpdate > 10)
            {
                learningRate *= 0.75;
                learner.ResetLearningRate(new CNTK.TrainingParameterScheduleDouble(learningRate));
                slotSinceLastUpdate = 0;
                should_stop = (learningRate < 1e-6);
            }
            return should_stop;
        }
    }
}
