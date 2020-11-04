using CoinRecognitionExample.Detection;
using CoinRecognitionExample.Recognition;
using Python.Runtime;
using System;

namespace CoinRecognitionExample
{
    class Program
    {
        static void Main(string[] args)
        {
            string filePath = @"C:/Users/arnal/Documents/coins.jpg";
            var coinDetector = new CoinDetector(filePath);
            coinDetector.ImagePreprocessing();

            var numberClasses = 60;
            var fileExt = new string[] { ".png" };
            var dataSetFilePath = @"C:/Users/arnal/Downloads/coin_dataset";
            var predictImgPath = dataSetFilePath + "/" + "class6_image1.png";

            var dataSet = new PreProcessing.DataSet(dataSetFilePath, fileExt, numberClasses, 0.2);
            dataSet.LoadDataSet();

            var cnn = new Cnn(dataSet);
            cnn.Train();
            //Console.WriteLine("Predicted: " + cnn.Predict(predictImgPath));
            
            Console.ReadLine();
        }

        private static void SetupPyEnv()
        {
            string envPythonHome = @"C:\Users\arnal\AppData\Local\Programs\Python\Python37\";
            string envPythonLib = envPythonHome + "Lib\\;" + envPythonHome + @"Lib\site-packages\";
            Environment.SetEnvironmentVariable("PYTHONHOME", envPythonHome, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PATH", envPythonHome + ";" + envPythonLib + ";" + Environment.GetEnvironmentVariable("PATH", EnvironmentVariableTarget.Machine), EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PYTHONPATH", envPythonLib, EnvironmentVariableTarget.User);
           
            PythonEngine.PythonHome = envPythonHome;
            PythonEngine.PythonPath = Environment.GetEnvironmentVariable("PYTHONPATH");
        }
    }
}
