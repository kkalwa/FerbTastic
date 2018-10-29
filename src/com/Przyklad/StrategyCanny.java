package com.Przyklad;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class StrategyCanny extends Strategy {

    @Override
    public double computeScore(Compose launcher) {
        Mat gray = launcher.grayscale;
        Mat input = launcher.image;
        Mat output = new Mat();

        double[] size = {0.0};
        MatOfPoint bullseye = launcher.findBullsEye(input, output, size);
        MatOfPoint2f polyBulls = new MatOfPoint2f();
        MatOfPoint2f converted = new MatOfPoint2f(bullseye.toArray());
        Imgproc.approxPolyDP(converted, polyBulls, 1, true);
        Rect bullRect = Imgproc.boundingRect(bullseye);
        Point center = new Point();
        center.x = (bullRect.br().x + bullRect.tl().x) * 0.5;
        center.y = (bullRect.br().y + bullRect.tl().y) * 0.5;
        bullseye = new MatOfPoint(polyBulls.toArray());
        List<Point> bullList = bullseye.toList();
        List<Point> promising = FerbTastic.analyzePoints(bullList, center);
        bullseye.fromList(promising);
        bullRect = Imgproc.boundingRect(bullseye);
        center.x = (bullRect.br().x + bullRect.tl().x) * 0.5;
        center.y = (bullRect.br().y + bullRect.tl().y) * 0.5;

        Imgproc.medianBlur(gray, gray, 5);
        Core.MinMaxLocResult result = Core.minMaxLoc(gray);
        System.out.println("Result: " + result.maxVal);
        double max = 50;
        double min = max/2;
        Mat canny = new Mat();
        Imgproc.Canny(gray, canny, min, max);

        Imgproc.morphologyEx(canny, canny, Imgproc.MORPH_DILATE, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3)));
        FerbTastic.saveFile(canny, "canny.jpg");

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(canny, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        System.out.println("Znaleziono: " + contours.size() + " konturow");
        int radius = 100;
        for(MatOfPoint e: contours)
        {
            promising = FerbTastic.analyzePoints(e.toList(), center);
            if(promising.size() > 50) {

                RotatedRect rect = launcher.simpleEllipse(e);
                //Imgproc.drawMarker(input, rect.center, new Scalar(0, 0, 255));
                if(launcher.euclidDst(center, rect.center) < radius) {
                    Imgproc.ellipse(launcher.image, rect, new Scalar(0, 0, 255), 3);
                }

            }
        }
        Imgproc.circle(launcher.image, center, radius, new Scalar(255,0,0), 3);
        FerbTastic.saveFile(launcher.image, "Srodki.jpg");
        return 0;
    }
}
