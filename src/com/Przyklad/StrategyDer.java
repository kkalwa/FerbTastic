package com.Przyklad;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class StrategyDer extends Strategy {
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

        Mat scharr = FerbTastic.ScharrDer(gray);
        FerbTastic.saveFile(scharr,"Scharr.jpg");
        Mat otsu = new Mat();
        FerbTastic.stdOtsu(scharr, otsu);
        FerbTastic.saveFile(otsu, "scharr_otsu.jpg");

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(otsu, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
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
