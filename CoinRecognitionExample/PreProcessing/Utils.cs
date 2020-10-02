using CoinRecognitionExample.Config;
using Keras.PreProcessing.Image;
using Numpy;

namespace CoinRecognitionExample.PreProcessing
{
    public class Utils
    {
        public static NDarray Normalize(string path)
        {
            var colorMode = Settings.Channels == 3 ? "rgb" : "grayscale";
            var img = ImageUtil.LoadImg(path, color_mode: colorMode, target_size: (Settings.ImgWidth, Settings.ImgHeight));
            return ImageUtil.ImageToArray(img) / 255;
        }

    }
}
