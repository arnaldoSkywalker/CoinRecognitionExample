using CoinRecognitionExample.Detection;
using CoinRecognitionExample.Recognition;
using Python.Included;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CoinRecognitionExample
{
    class Program
    {
        static void Main(string[] args)
        {
            string filePath = @"C:/Users/arnal/Documents/coins.jpg";
            var coinDetector = new CoinDetector(filePath);
            // coinDetector.ImagePreprocessing();

            //SetupPyEnv();

            var numberClasses = 60;
            var fileExt = new string[] { ".png" };
            var dataSetFilePath = @"C:/Users/arnal/Downloads/coin_dataset";
            var dataSet = new PreProcessing.DataSet(dataSetFilePath, fileExt, numberClasses, 0.2);
            dataSet.LoadDataSet();

            var cnn = new Cnn(dataSet);
            cnn.Train();

            var preTrainedCnn = new PreTrainedCnn();
            //preTrainedCnn.Train(dataSet);

            Console.ReadLine();
        }

        private static void SetupPyEnv()
        {
            string envPythonHome = @"C:\Users\arnal\AppData\Local\Programs\Python\Python37\";
            string envPythonLib = envPythonHome + "Lib\\;" + envPythonHome + @"Lib\site-packages\";
            Environment.SetEnvironmentVariable("PYTHONHOME", envPythonHome, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PATH", envPythonHome + ";" + envPythonLib + ";" + Environment.GetEnvironmentVariable("PATH", EnvironmentVariableTarget.Machine), EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PYTHONPATH", envPythonLib, EnvironmentVariableTarget.Process);
           
            PythonEngine.PythonHome = envPythonHome;
            PythonEngine.PythonPath = Environment.GetEnvironmentVariable("PYTHONPATH");
        }
    }
}
