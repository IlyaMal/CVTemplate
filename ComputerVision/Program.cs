using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Drawing;

class Program
{
    static void Main(string[] args)
    {
        VideoCapture capture = new VideoCapture(0);
        if (!capture.IsOpened)
        {
            Console.WriteLine("Не удалось открыть веб-камеру.");
            return;
        }

        CascadeClassifier faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
        
        Mat frame = new Mat();
        while (true)
        {
            capture.Read(frame);
            if (frame.IsEmpty)
            {
                break;
            }

            Mat grayFrame = new Mat();
            CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);

            Rectangle[] faces = faceCascade.DetectMultiScale(grayFrame, 1.1, 10, Size.Empty, Size.Empty);
            foreach (Rectangle face in faces)
            {
                CvInvoke.Rectangle(frame, face, new MCvScalar(0, 255, 0), 2);
            }
            
            CvInvoke.Imshow("Webcam Face Detection", frame);

            if (CvInvoke.WaitKey(1) == 27) // 27 - код клавиши 'Esc'
            {
                break;
            }
        }


        capture.Release();
        CvInvoke.DestroyAllWindows();
    }
}