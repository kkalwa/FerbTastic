package com.Przyklad;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class StrategyPer extends Strategy {
    @Override
    public double computeScore(Compose launcher) {
        Mat gray = launcher.grayscale;
        Mat input = launcher.image;
        Mat output = new Mat();

        /*double x = input.cols() / 147.0, y = input.rows() / 196.0; // dane do rozmazania
        if( (int)(x%2) == 0) // jesli x jest parzyste to nalezy zmniejszyc o 1
            x = x-1;
        if( ((int)y % 2) == 0) // jesli y parzysty to zmniejszamy o 1
            y = y-1;
        System.out.println(x + "\n" + y);
        //Imgproc.GaussianBlur(gray, gray, new Size(x, y), 5); // Na poczatku nastepuje mocne rozmycie obrazu
        //FerbTastic.saveFile(gray, FerbTastic.counter+"rozmyty.jpg");
        //FerbTastic.saveFile(gray, "blurred.jpg");
        List<RotatedRect> target = new ArrayList<RotatedRect>(); // Lista elips, ktore tworza tarcze
        double[] size = {0.0}; // Rozmiar bullseye, potrzebny do okreslania odleglosci na tarczy. Jest tablica aby mozna bylo go przeslac jako argument - wynik
        MatOfPoint bullseye = launcher.findBullsEye(input, output, size); // przeszuka obraz w poszukiwaniu bullseye i zwroci kontur go reprezentujacy oraz jako argument-wynik binarke oraz rozmiar
        System.out.println("Rozmiar: " + size[0]);
        MatOfPoint2f polyBulls = new MatOfPoint2f();
        MatOfPoint2f converted = new MatOfPoint2f(bullseye.toArray());
        Imgproc.approxPolyDP(converted, polyBulls, 1, true);
        Rect bullRect = Imgproc.boundingRect(bullseye);
        Point center = new Point(); // Punkt reprezentujacy centrum tarczy. W trakcie dzialania programu bedzie wiele razy modyfikowany
        center.x = (bullRect.br().x + bullRect.tl().x) * 0.5;
        center.y = (bullRect.br().y + bullRect.tl().y) * 0.5;
        bullseye = new MatOfPoint(polyBulls.toArray());
        List<Point> bullList = bullseye.toList();
        List<Point> promising = Compose.analyzePoints(bullList, center, true);
        bullseye.fromList(promising);
        bullRect = Imgproc.boundingRect(bullseye);
        center.x = (bullRect.br().x + bullRect.tl().x) * 0.5;
        center.y = (bullRect.br().y + bullRect.tl().y) * 0.5;

        Mat wzorzec = FerbTastic.openFile("WZORCE1.jpg");
        MatOfPoint bullseye2 = launcher.findBullsEye(wzorzec, output, size);
        Imgproc.rectangle(input,bullRect.tl(), bullRect.br(), new Scalar(0,0,255), 1);
        Rect rect = Imgproc.boundingRect(bullseye2);
        Imgproc.rectangle(wzorzec,rect.tl(), rect.br(), new Scalar(0,0,255), 1);
        Point myPoint = new Point(500,500);
        Imgproc.drawMarker(input, myPoint, new Scalar(0,0,255));
        FerbTastic.saveFile(input, "znowTesty.jpg");
        FerbTastic.saveFile(wzorzec, "wzor.jpg");
        List<Point> p1 = new ArrayList<Point>();
        List<Point> p2 = new ArrayList<Point>();
        Point tl = bullRect.tl();
        Point br = bullRect.br();
        p1.add(tl);
        p1.add(new Point(br.x, tl.y));
        p1.add(new Point(tl.x, br.y));
        p1.add(br);

        tl = rect.tl();
        br = rect.br();
        p2.add(tl);
        p2.add(new Point(br.x, tl.y));
        p2.add(new Point(tl.x, br.y));
        p2.add(br);

        MatOfPoint2f orginal = new MatOfPoint2f();
        orginal.fromList(p1);
        MatOfPoint2f pattern = new MatOfPoint2f();
        pattern.fromList(p2);
        System.out.println(orginal);
        System.out.println(pattern);
        Mat M = Imgproc.getPerspectiveTransform(orginal, pattern);
        p1.clear();
        p1.add(myPoint);
        pattern.fromList(p1);
        Mat perspective = new Mat();
        Imgproc.warpPerspective(input, perspective, M, new Size(512, 512));
        FerbTastic.saveFile(perspective, "per.jpg");*/
        List<Point> patternList = generateTarget(1024, 768, 300);
        double[] size = {0.0};

        Mat brightest = new Mat();
        input = FerbTastic.openFile("S40.jpg");
        input = Resolution.changeRes(input, 768, 1024);
        Compose.toGray(input, gray);

        Imgproc.GaussianBlur(gray, gray, new Size(15,15), 5.0,5.0);

        MatOfPoint bullseye = Compose.findBullsEye(input, new Mat(), size);
        Compose.stdOtsu(gray, gray);
        FerbTastic.saveFile(gray, "stdOtsu.jpg");
        List<MatOfPoint> contours = extractContours(gray);
        distances(bullseye, contours);
        Imgproc.drawContours(input, contours, -1, new Scalar(0,0,255));
        FerbTastic.saveFile(input, "input.jpg");
        //numbers(contours);
        Rect rect = Imgproc.boundingRect(bullseye);

        FerbTastic.toGray(input, gray);
        Point center = new Point();
        center.x = (rect.br().x + rect.tl().x) * 0.5;
        center.y = (rect.br().y + rect.tl().y) * 0.5;
        Imgproc.threshold(gray,brightest,215, 255, Imgproc.THRESH_BINARY);
        //FerbTastic.toGray(brightest, brightest);
        Imgproc.morphologyEx(brightest, brightest, Imgproc.MORPH_DILATE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7,7)));
        Imgproc.morphologyEx(brightest, brightest, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(15,15)));
        FerbTastic.saveFile(brightest, "bright.jpg");
        contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(brightest, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        List<Point> closest = StrategyPer.findClosestPoints2(contours, center);
        FerbTastic.drawPoints(input, closest);
        FerbTastic.saveFile(input, "O2.jpg");

        MatOfPoint2f patternMat = new MatOfPoint2f();
        patternMat.fromList(patternList);


        List<Point> realList = new ArrayList<Point>();
        Imgproc.drawMarker(input, new Point(383,263), new Scalar(0,0,255));
        Imgproc.drawMarker(input, new Point(378,568), new Scalar(0,0,255));
        Imgproc.drawMarker(input, new Point(227,419), new Scalar(0,0,255));
        Imgproc.drawMarker(input, new Point(534,424), new Scalar(0,0,255));
        FerbTastic.saveFile(input, "testy.jpg");
        realList.add(new Point(383,263));
        realList.add(new Point(378,568));
        realList.add(new Point(227,419));
        realList.add(new Point(534,424));
        /*realList.add(rect.tl());
        realList.add(new Point(rect.br().x, rect.tl().y));
        realList.add(rect.br());
        realList.add(new Point(rect.tl().x, rect.br().y));*/

        MatOfPoint2f realMat = new MatOfPoint2f();
        realMat.fromList(realList);
        Mat M = Imgproc.getPerspectiveTransform(realMat, patternMat);

        Mat brightest2 = new Mat();
        Imgproc.warpPerspective(brightest, brightest2, M, new Size(input.cols(),input.rows()));
        FerbTastic.saveFile(brightest2, "bright2.jpg");
        //Mat rescaled = new Mat();
        //Imgproc.warpPerspective(input, rescaled, M, new Size(768, 1024), Imgproc.INTER_LANCZOS4);
        //FerbTastic.saveFile(rescaled, "rescaled.jpg");

        Mat pattern = FerbTastic.openFile("drew.jpg");
        pattern.setTo(new Scalar(0,0,255), brightest2);
        FerbTastic.saveFile(pattern, "zlozenie.jpg");
        return 0;
    }
    public static void distances(MatOfPoint bullseye, List<MatOfPoint> contours)
    {
        MatOfPoint2f curve = new MatOfPoint2f();
        Imgproc.approxPolyDP(new MatOfPoint2f(bullseye.toArray()), curve, 0.1, true);
        List<Double> dist = new ArrayList<Double>();
        for(int i=0; i<contours.size(); ++i)
        {
            MatOfPoint current = contours.get(i);
            Rect rect = Imgproc.boundingRect(current);

            Point center = new Point();
            center.x = (rect.br().x + rect.tl().x) * 0.5;
            center.y = (rect.br().y + rect.tl().y) * 0.5;
            System.out.println(i+": "+Imgproc.pointPolygonTest(curve, center, true));
            if(Imgproc.pointPolygonTest(curve, center, true) > 0.0)
            {
                dist.add(Imgproc.pointPolygonTest(curve, center, true));
            }
        }

    }
    public static void numbers(List<MatOfPoint> contours)
    {
        for(MatOfPoint c: contours) {
            System.out.println(Imgproc.boundingRect(c).size().width + "\t" + Imgproc.boundingRect(c).size().height);
        }
    }
    public static List<MatOfPoint> extractContours(Mat image)
    {
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(image, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
        return contours;
    }
List<Double> findClosestPoints(List<MatOfPoint> contours, Point center)
{
    List<Point> output = new ArrayList<Point>();
    List<Double> outputList = new ArrayList<Double>();
    for(MatOfPoint c: contours)
    {
        List<Double> distances = Compose.euclidDist(c.toList(), center);
        outputList.add(Collections.min(distances));
    }
    return outputList;
}
    List<Point> detectNumbers(Mat gray)
    {
        Mat otsu = new Mat();
        Compose.stdOtsu(gray, otsu);
        FerbTastic.saveFile(otsu, "otsuCenters.jpg");
        return null;
    }
    public static List<Point> findClosestPoints2(List<MatOfPoint> contours, Point center)
    {
        List<Point> output = new ArrayList<Point>();
        for(int i=0; i<contours.size(); ++i)
        {
            List<Double> distances = Compose.euclidDist(contours.get(i).toList(), center);
            int minIndex = 0;
            double min = Collections.min(distances);
            for(int j=1; j<distances.size(); ++j)
            {
                double dist = distances.get(j);
                if(dist == min)
                {
                    output.add(contours.get(i).toList().get(j));
                    break;
                }
            }
        }
        return output;
    }

    public static List<Point> generateTarget(int rows, int cols, int diameter)
    {
        Mat target = new Mat(rows, cols, CvType.CV_8UC1);
        target.setTo(Scalar.all(255));
        Point center = new Point(cols/2, rows/2);
        double ratio = (diameter)/49.0;
        Imgproc.circle(target, center, (int)(diameter/2), Scalar.all(0), -1);

        int radius = (int)(diameter/2);

        for(int i=1; i<=6; ++i)
        {
            int r = (int)(radius + i*6.0*ratio);
            Imgproc.circle(target, center, r, Scalar.all(0), 1);
        }

        for(int i=1; i<=3; ++i)
        {
            int r = (int)(radius - i*6.0*ratio);
            Imgproc.circle(target, center, r, Scalar.all(255), 1);
        }
        List<Point> points = new ArrayList<Point>();
        /*points.add(new Point(center.x - radius, center.y - radius));
        points.add(new Point(center.x + radius, center.y - radius));
        points.add(new Point(center.x + radius, center.y + radius));
        points.add(new Point(center.x - radius, center.y + radius));*/
        Imgproc.drawMarker(target, new Point(center.x, center.y - ratio*27), Scalar.all(0));
        Imgproc.drawMarker(target, new Point(center.x, center.y + ratio*27), Scalar.all(0));
        Imgproc.drawMarker(target, new Point(center.x - ratio*27, center.y), Scalar.all(0));
        Imgproc.drawMarker(target, new Point(center.x + ratio*27, center.y), Scalar.all(0));
        points.add(new Point(center.x, center.y - ratio*27));
        points.add(new Point(center.x, center.y + ratio*27));
        points.add(new Point(center.x - ratio*27, center.y));
        points.add(new Point(center.x + ratio*27, center.y));
        FerbTastic.saveFile(target, "drew.jpg");
        return points;
    }
}
