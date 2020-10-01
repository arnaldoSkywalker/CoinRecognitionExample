using OpenCvSharp;

namespace CoinRecognitionExample.Detection
{
    public class CoinDetector
    {
        private Mat _image;
        private Mat _originalImage;
        private string _pathToFile;

        public CoinDetector(string pathToFile)
        {
            _pathToFile = pathToFile;
        }

        public void ImagePreprocessing()
        {
            _image = new Mat(_pathToFile, ImreadModes.Color);
            _originalImage = _image.Clone();
            TransformGrayScale();
            TransformGaussianBlur();
            HoughSegmentation();
        }

        private void TransformGrayScale()
        {
            _image = _originalImage.CvtColor(ColorConversionCodes.BGR2GRAY);
            new Window("Grayed Coins", WindowMode.Normal, _image);
            //Cv2.WaitKey();
        }

        private void TransformGaussianBlur()
        {
            Cv2.GaussianBlur(_image, _image, new Size(0, 0), 1);
            new Window("Blurred Coins", WindowMode.Normal, _image);
            //Cv2.WaitKey();
        }

        private void HoughSegmentation()
        {
            Mat result = _image.Clone();

            var circleSegments = Cv2.HoughCircles(_image, HoughMethods.Gradient, 1.02, 40);
            for (int i = 0; i < circleSegments.Length; i++)
            {
                Cv2.Circle(result, (Point) circleSegments[i].Center, (int)circleSegments[i].Radius, new Scalar(255, 255, 0), 2);
            }

            using (new Window("Circles", result))
            {
                Cv2.WaitKey();
            }
        }
    }
}
