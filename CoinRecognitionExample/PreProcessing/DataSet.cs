using Keras.Utils;
using Numpy;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CoinRecognitionExample.PreProcessing
{
    public class DataSet
    {
        private string _pathToFolder;
        private string[] _extList;
        private List<int> _labels;
        private List<NDarray> _objs;
        private double _validationSplit;
        public int NumberClasses { get; set; }
        public NDarray TrainX { get; set; }
        public NDarray ValidationX { get; set; }
        public NDarray TrainY { get; set; }
        public NDarray ValidationY { get; set; }

        public DataSet(string pathToFolder, string[] extList, int numberClasses, double validationSplit)
        {
            _pathToFolder = pathToFolder;
            _extList = extList;
            NumberClasses = numberClasses;
            _labels = new List<int>();
            _objs = new List<NDarray>();
            _validationSplit = validationSplit;
        }

        public void LoadDataSet()
        {
            // Process the list of files found in the directory.
            string[] fileEntries = Directory.GetFiles(_pathToFolder);
            foreach (string fileName in fileEntries)
                ProcessFile(fileName);

            MapToClassRange();
            GetTrainValidationData();
        }

        private void MapToClassRange()
        {
            HashSet<int> uniqueLabels = _labels.ToHashSet();
            var uniqueLabelList = uniqueLabels.ToList();
            uniqueLabelList.Sort();

            _labels = _labels.Select(x => uniqueLabelList.IndexOf(x)).ToList();
        }

        private NDarray OneHotEncoding(List<int> labels)
        {
            var npLabels = np.array(labels.ToArray()).reshape(-1);
            return Util.ToCategorical(npLabels, num_classes: NumberClasses);
        }

        private void ProcessFile(string path)
        {
            _objs.Add(Utils.Normalize(path));
            ProcessLabel(Path.GetFileName(path));
        }

        private void ProcessLabel(string filename)
        {
            _labels.Add(int.Parse(ExtractClassFromFileName(filename)));
        }

        private string ExtractClassFromFileName(string filename)
        {
            return filename.Split('_')[0].Replace("class", "");
        }

        private void GetTrainValidationData()
        {
            var listIndices = Enumerable.Range(0, _labels.Count).ToList();
            var toValidate = _objs.Count * _validationSplit;
            var random = new Random();
            var xValResult = new List<NDarray>();
            var yValResult = new List<int>();
            var xTrainResult = new List<NDarray>();
            var yTrainResult = new List<int>();

            // Split validation data
            for (var i = 0; i < toValidate; i++)
            {
                var randomIndex = random.Next(0, listIndices.Count);
                var indexVal = listIndices[randomIndex];
                xValResult.Add(_objs[indexVal]);
                yValResult.Add(_labels[indexVal]);
                listIndices.RemoveAt(randomIndex);
            }

            // Split rest (training data)
            listIndices.ForEach(indexVal => 
            { 
                xTrainResult.Add(_objs[indexVal]);
                yTrainResult.Add(_labels[indexVal]);
            });

            TrainY = OneHotEncoding(yTrainResult);
            ValidationY = OneHotEncoding(yValResult);
            TrainX = np.array(xTrainResult);
            ValidationX = np.array(xValResult);
        }
    }
}
