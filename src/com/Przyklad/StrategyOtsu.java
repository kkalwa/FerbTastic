package com.Przyklad;

import org.opencv.core.*;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class StrategyOtsu extends Strategy {
   void threshold(Mat input, Mat output, int threshold)
   {
       Imgproc.threshold(input, output, threshold, 255, Imgproc.THRESH_BINARY);
   }
    @Override
    public double computeScore(Compose launcher) {
        Mat input = launcher.image;
        Mat gray = launcher.grayscale;
        gray = Resolution.changeRes(gray, 1000, 1000);
        //Imgproc.GaussianBlur(gray, gray, new Size(11,11), 5, 5);
        //Imgproc.equalizeHist(gray, gray);

        CLAHE clahe = Imgproc.createCLAHE(3, new Size(10,10));
        clahe.apply(gray, gray);
        FerbTastic.saveFile(gray, "histogram/"+FerbTastic.counter+"histogram.jpg");

        return 0;
    }
}
