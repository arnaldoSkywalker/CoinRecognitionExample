using CoinRecognitionExample.Config;
using CoinRecognitionExample.PreProcessing;
using Keras;
using Keras.Layers;
using Keras.Models;
using Numpy;
using System;


namespace CoinRecognitionExample.Recognition
{
    public class Cnn
    {
        private DataSet _dataset;
        private Sequential _model;

        public Cnn(DataSet dataset)
        {
            _dataset = dataset;
            _model = new Sequential();
        }

        public void Train()
        {
            // Build CNN model
            _model.Add(new Conv2D(32, kernel_size: (3, 3).ToTuple(),
                                 padding: Settings.PaddingMode,
                                 input_shape: new Shape(Settings.ImgWidth, Settings.ImgHeight, Settings.Channels)));
            _model.Add(new Activation(Settings.ActivationFunction));
            _model.Add(new Conv2D(32, (3, 3).ToTuple()));
            _model.Add(new Activation(Settings.ActivationFunction));
            _model.Add(new MaxPooling2D(pool_size: (2, 2).ToTuple()));
            _model.Add(new Dropout(0.25));

            _model.Add(new Conv2D(64, kernel_size: (3, 3).ToTuple(),
                                padding: Settings.PaddingMode));
            _model.Add(new Activation(Settings.ActivationFunction));
            _model.Add(new Conv2D(64, (3, 3).ToTuple()));
            _model.Add(new Activation(Settings.ActivationFunction));
            _model.Add(new MaxPooling2D(pool_size: (2, 2).ToTuple()));
            _model.Add(new Dropout(0.25));

            _model.Add(new Flatten());
            _model.Add(new Dense(Settings.FullyConnectedNodes));
            _model.Add(new Activation(Settings.ActivationFunction));
            _model.Add(new Dropout(0.5));
            _model.Add(new Dense(_dataset.NumberClasses));
            _model.Add(new Softmax());
            
            _model.Compile(loss: Settings.LossFunction,
              optimizer: Settings.Optimizer, 
              metrics: new string[] { Settings.Accuracy });
            
            _model.Fit(_dataset.TrainX, _dataset.TrainY,
                          batch_size: Settings.BatchSize,
                          epochs: Settings.Epochs,
                          validation_data: new NDarray[] { _dataset.ValidationX, _dataset.ValidationY });

            var score = _model.Evaluate(_dataset.ValidationX, _dataset.ValidationY, verbose: 0);
            Console.WriteLine("Test loss:" + score[0]);
            Console.WriteLine("Test accuracy:" + score[1]);
        }

        public NDarray Predict(string imgPath)
        {
            return _model.Predict(Utils.Normalize(imgPath));
        }
    }
}
