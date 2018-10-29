package com.Przyklad;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class Resolution {
    public static Mat changeRes(Mat input, double x, double y)
    {
        Mat output = new Mat();
        double factorx = 0;
        double factory = 0;

        factorx = x / input.cols();
        factory = y / input.rows();
        Imgproc.resize(input, output, new Size(), factorx, factory, Imgproc.INTER_LANCZOS4); // Tutaj nastepuje redukcja rozdzielczosci
        return output;
    }
}
