using CoinRecognitionExample.Config;
using Keras.PreProcessing.Image;
using Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CoinRecognitionExample.PreProcessing
{
    public class Utils
    {
        public static NDarray Normalize(string path)
        {
            var img = ImageUtil.LoadImg(path, target_size: (Settings.ImgWidth, Settings.ImgHeight));
            return ImageUtil.ImageToArray(img) / 255;
        }

    }
}
