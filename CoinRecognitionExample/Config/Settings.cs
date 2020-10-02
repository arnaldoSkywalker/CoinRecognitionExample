using Keras;
using Keras.Optimizers;

namespace CoinRecognitionExample.Config
{
    public class Settings
    {
        public const int ImgWidth = 64;
        public const int ImgHeight = 64;
        public const int MaxValue = 255;
        public const int MinValue = 0;
        public const int Channels = 3;
        public const int BatchSize = 12;
        public const int Epochs = 100;
        public const int FullyConnectedNodes = 512;
        public const string LossFunction = "categorical_crossentropy";
        public const string Accuracy = "accuracy";
        public const string ActivationFunction = "relu";
        public const string PaddingMode = "same";
        public static StringOrInstance Optimizer = new RMSprop(lr: Lr, decay: Decay);
        private const float Lr = 0.0001f;
        private const float Decay = 1e-6f;
    }
}
