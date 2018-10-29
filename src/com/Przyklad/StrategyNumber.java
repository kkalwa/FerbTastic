package com.Przyklad;

import org.opencv.core.*;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class StrategyNumber extends Strategy {
    @Override
    public double computeScore(Compose launcher) {
        Mat input = launcher.image;
        Mat gray = launcher.grayscale;
        System.out.println("Iteracja: " + FerbTastic.counter);
        double[] size = {0.0};

        MatOfPoint bullseye = Compose.findBullsEye(input, new Mat(), size);
        Point bullCenter = Compose.computeCenter(bullseye);
        Mat mask = new Mat(gray.rows(), gray.cols(), CvType.CV_8UC1);
        mask.setTo(Scalar.all(0));
        Point center = new Point();
        float[] radius = {0.0f};
        Imgproc.minEnclosingCircle(new MatOfPoint2f(bullseye.toArray()), center, radius);
        radius[0] += 5;
        Imgproc.circle(mask, center, (int)radius[0], Scalar.all(255), -1);
        FerbTastic.saveFile(mask, "mask.jpg");
        List<MatOfPoint> auxiliary = new ArrayList<MatOfPoint>();
        Imgproc.findContours(mask, auxiliary, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        bullseye = auxiliary.get(0);
        MatOfPoint2f polyBull = new MatOfPoint2f();
        Imgproc.approxPolyDP(new MatOfPoint2f(bullseye.toArray()), polyBull, 1, true);



        Imgproc.GaussianBlur(gray, gray, new Size(5,5), 1.5, 1.5);
        Mat binarized = new Mat();
        Mat blackhat = new Mat();
        Imgproc.morphologyEx(gray, blackhat, Imgproc.MORPH_BLACKHAT, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7,7)));
        int clipLimit = 3;
        CLAHE clahe = Imgproc.createCLAHE(clipLimit, new Size(10,10));
        clahe.apply(blackhat, blackhat);
        FerbTastic.saveFile(gray, "histogram/"+FerbTastic.counter+"histogram.jpg");
        FerbTastic.saveFile(blackhat,"morphology/"+FerbTastic.counter+"blackhat.jpg");

        Imgproc.threshold(blackhat, binarized, 10, 255, Imgproc.THRESH_BINARY);
        Imgproc.morphologyEx(binarized, binarized, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3,3)));

        Mat brightest = Compose.extractBrightest(gray);
        Imgproc.morphologyEx(brightest, brightest, Imgproc.MORPH_DILATE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(17,17)));
        auxiliary.clear();
        Imgproc.findContours(brightest, auxiliary, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        List<MatOfPoint2f> polynomials = Compose.createPolys(auxiliary);
        binarized.setTo(Scalar.all(255), brightest);
        Imgproc.circle(binarized, center,(int)radius[0], Scalar.all(255), -1);
        FerbTastic.saveFile(binarized, "binarized/"+FerbTastic.counter+"binarized.jpg");
        FerbTastic.saveFile(binarized, FerbTastic.counter+"binarized.jpg");
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(binarized, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
        System.out.println("Znaleziono: " + contours.size() + " po binaryzacji");

        List<MatOfPoint> filtered = new ArrayList<MatOfPoint>();
        List<Double> distList = new ArrayList<Double>();
        double ratio = 49.0/size[0];
        for(int i=0; i<contours.size(); ++i)
        {
            MatOfPoint current = contours.get(i);
            Rect rect = Imgproc.boundingRect(current);
            Point c = new Point();
            c.x = (rect.br().x + rect.tl().x) * 0.5;
            c.y = (rect.br().y + rect.tl().y) * 0.5;
            double dist = Imgproc.pointPolygonTest(polyBull, c, true);
            if(  rect.height > 10 && rect.width > 5 && dist < 0.0 && Math.abs(dist*ratio) < 37 && ((double)rect.width/(double)rect.height) < 1.0 &&
                    rect.height < 0.05*input.rows() && rect.width < 0.05*input.cols() && isHole(polynomials, c) != true)
            {
                //classifyCont(current);
                filtered.add(current);
                distList.add(Math.abs(dist*ratio));
                System.out.println("Szerokosc na wysokosc: " + ((double)rect.width/(double)rect.height));
            }
        }
        List<Integer> groups = group(filtered);
        Mat colors = new Mat(gray.rows(), gray.cols(), CvType.CV_8UC3);
        colors.setTo(Scalar.all(0));
        colorGroups(colors, filtered, groups); // Tylko kolorowanie
        List<MatOfPoint> A = new ArrayList<MatOfPoint>();
        List<MatOfPoint> B = new ArrayList<MatOfPoint>();
        List<MatOfPoint> C = new ArrayList<MatOfPoint>();
        List<MatOfPoint> D = new ArrayList<MatOfPoint>();
        splitGroup(filtered, groups, A, B, C, D);
        Point realCenter = drawLines(A,B,C,D,gray.rows(), gray.cols());
        Point top = new Point();
        Point right = new Point();
        Point bottom = new Point();
        Point left = new Point();
        pickClosestPoints(A,B,C,D, realCenter,top, right, bottom, left, gray.rows(), gray.cols()); // zwraca najblizsze punkty i klasyfikuje je jako top, right, left i bottom
        List<Integer> numbers = new ArrayList<Integer>();
        whichNumbers(realCenter, top, right, bottom, left, numbers, size[0]);
        Mat mask2 = new Mat(gray.rows(), gray.cols(), CvType.CV_8UC1);
        mask2.setTo(Scalar.all(0));
        for(int i=0; i<numbers.size(); ++i)
        {
            int a = numbers.get(i);
            Imgproc.putText(mask2, Integer.toString(a), new Point(20 + i*10,20 + i*10), Core.FONT_HERSHEY_SIMPLEX, 1.0, Scalar.all(255));
        }
        FerbTastic.saveFile(mask2, "test/"+FerbTastic.counter+"test.jpg");

        Mat filter = new Mat(gray.rows(), gray.cols(), CvType.CV_8UC3);
        filter.setTo(Scalar.all(0));
        Imgproc.cvtColor(binarized, binarized, Imgproc.COLOR_GRAY2BGR);
        Imgproc.drawContours(filter, filtered, -1, new Scalar(0,0,255), -1);
        FerbTastic.saveFile(filter, "filtered/"+FerbTastic.counter + "filtered.jpg");
        List<Integer> indexes = new ArrayList<Integer>();
        double max = Collections.max(distList);
        Mat maxes = new Mat(gray.rows(), gray.cols(), CvType.CV_8UC3);
        for(int i=0;i<4;++i)
        {
            double min = Collections.min(distList);
            for(int j=0; i<distList.size(); ++j)
            {
                if(distList.get(j) == min)
                {
                    indexes.add(j);
                    distList.add(j,max);
                    distList.remove(j+1);
                    break;
                }
            }
        }
        List<Rect> six = new ArrayList<Rect>();
        six.add(Imgproc.boundingRect(filtered.get(indexes.get(0))));
        six.add(Imgproc.boundingRect(filtered.get(indexes.get(1))));
        six.add(Imgproc.boundingRect(filtered.get(indexes.get(2))));
        six.add(Imgproc.boundingRect(filtered.get(indexes.get(3))));

        int minX = 0, minY = 0;
        int maxX = 0, maxY = 0;
        double cXmin = gray.cols(), cYmin = gray.cols();
        double cXmax = 0.0, cYmax = 0.0;
        Point pXmin = new Point();
        Point pXmax = new Point();
        Point pYmin = new Point();
        Point pYmax = new Point();
        for(int i=0; i<six.size(); ++i) {
            Rect current = six.get(i);
            center.x = (current.br().x + current.tl().x) * 0.5;
            center.y = (current.br().y + current.tl().y) * 0.5;
            if(center.x < cXmin)
            {
                minX = i;
                cXmin = center.x;
                pXmin.x = center.x;
                pXmin.y = center.y;
            }
             if(center.x > cXmax)
            {
                maxX = i;
                cXmax = center.x;
                pXmax.x = center.x;
                pXmax.y = center.y;
            }
            if(center.y > cYmax)
            {
                maxY = i;
                cYmax = center.y;
                pYmax.x = center.x;
                pYmax.y = center.y;
            }
             if(center.y < cYmin)
            {
                minY = i;
                cYmin = center.y;
                pYmin.x = center.x;
                pYmin.y = center.y;
            }

        }
        List<Point> rightOrder = new ArrayList<Point>();
        rightOrder.add(pYmin);
        rightOrder.add(pYmax);
        rightOrder.add(pXmin);
        rightOrder.add(pXmax);
        Imgproc.drawMarker(gray, rightOrder.get(0), Scalar.all(0));
        Imgproc.drawMarker(gray, rightOrder.get(1), Scalar.all(0));
        Imgproc.drawMarker(gray, rightOrder.get(2), Scalar.all(0));
        Imgproc.drawMarker(gray, rightOrder.get(3), Scalar.all(0));
        FerbTastic.saveFile(gray, FerbTastic.counter+"punkty.jpg");
        MatOfPoint2f realMat = new MatOfPoint2f();
        realMat.fromList(rightOrder);
        MatOfPoint2f patternMat = new MatOfPoint2f();
        patternMat.fromList(StrategyPer.generateTarget(gray.rows(), gray.cols(), (int)size[0]));
        Mat M = Imgproc.getPerspectiveTransform(realMat, patternMat);
        FerbTastic.saveFile(brightest, "brightest/"+FerbTastic.counter+"brightest.jpg");
        Imgproc.morphologyEx(brightest, brightest, Imgproc.MORPH_ERODE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(17,17)));
        Mat brightest2 = new Mat();
        Imgproc.warpPerspective(brightest, brightest2, M, new Size(input.cols(),input.rows()));
        FerbTastic.saveFile(brightest2, "brightest2/"+FerbTastic.counter+"brightest2.jpg");
        Mat pattern = FerbTastic.openFile("../wynik/drew.jpg");
        pattern.setTo(new Scalar(0,0,255), brightest2);
        FerbTastic.saveFile(pattern, "zlozenie/"+FerbTastic.counter+"zlozenie.jpg");
        FerbTastic.saveFile(gray, "zlozenie/"+FerbTastic.counter+"zlozenie0.jpg");

        return 0;
    }

    /**
     * Okresla ktora liczba zostala zaklasyfikowana jako najblizsza bullseye
     * @param realCenter
     * @param top
     * @param right
     * @param bottom
     * @param left
     * @param v
     */
    private void whichNumbers(Point realCenter, Point top, Point right, Point bottom, Point left, List<Integer> list, double v)
    {
        double ratio = 49.0/v;
        double topL = Compose.euclidDst(realCenter, top)*ratio;
        double rightL = Compose.euclidDst(realCenter, right)*ratio;
        double bottomL = Compose.euclidDst(realCenter, bottom)*ratio;
        double leftL = Compose.euclidDst(realCenter, left)*ratio; // Tutaj oblicza sie odleglosci od srodka (w milimetrach)

        double first = 28;
        List<Double> distances = new ArrayList<Double>();
        distances.add(first);
        int counter = 0;
        int base = 6;
        boolean topB = false;
        boolean rightB = false;
        boolean bottomB = false;
        boolean leftB = false;
        int topP = -1;
        int rightP = -1;
        int bottomP = -1;
        int leftP = -1;
        for(int i=1;i<=5;++i)
        {
            double dist = first + (double)i*6.7;
            distances.add(dist);
            if( topB == false && topL <= dist)
            {
                topP = base;
                topB = true;
                ++counter;
            }
            if( rightB == false && rightL <= dist)
            {
                rightP = base;
                rightB = true;
                ++counter;
            }
            if( bottomB == false && bottomL <= dist)
            {
                bottomP = base;
                bottomB = true;
                ++counter;
            }
            if( leftB == false && leftL <= dist)
            {
                leftP = base;
                leftB = true;
                ++counter;
            }
            --base;

            if(base <=0 || counter == 4)
            {
                break;
            }
        }
        list.add(topP);
        list.add(rightP);
        list.add(bottomP);
        list.add(leftP);
    }

    /**
     * Funkcja przeszuka znalezione cyfry by znalezc te znajdujace sie najblizej srodka
     * @param a - jedna z czterech grup konturow
     * @param b - jedna z czterech grup konturow
     * @param c - jedna z czterech grup konturow
     * @param d - jedna z czterech grup konturow
     * @param realCenter - srodek tarczy
     * @param top - punkt w ktorym zapisuje sie cyfre znajdujaca sie nad bullseye
     * @param right - punkt na prawo od bullseye
     * @param bottom - punkt ponizej bullseye
     * @param left - punkt na lewo od bullseye
     */
    private void pickClosestPoints(List<MatOfPoint> a, List<MatOfPoint> b, List<MatOfPoint> c,
                                   List<MatOfPoint> d, Point realCenter, Point top, Point right,
                                   Point bottom, Point left, int rows, int cols)
    {
        List<Point> centersA = Compose.computeCenter(a);
        List<Point> centersB = Compose.computeCenter(b);
        List<Point> centersC = Compose.computeCenter(c);
        List<Point> centersD = Compose.computeCenter(d);
        int minIndex = Compose.indexOfMin(Compose.euclidDist(centersA, realCenter));
        Point minA = centersA.get(minIndex);
        minIndex = Compose.indexOfMin(Compose.euclidDist(centersB, realCenter));
        Point minB = centersB.get(minIndex);
        minIndex = Compose.indexOfMin(Compose.euclidDist(centersC, realCenter));
        Point minC = centersC.get(minIndex);
        minIndex = Compose.indexOfMin(Compose.euclidDist(centersD, realCenter));
        Point minD = centersD.get(minIndex);
        List<Point> elements = new ArrayList<Point>();

        elements.add(minA);
        elements.add(minB);
        elements.add(minC);
        elements.add(minD);

        int minX = 0, minY = 0;
        int maxX = 0, maxY = 0;
        double cXmin = cols, cYmin = cols;
        double cXmax = 0.0, cYmax = 0.0;
        Point pXmin = new Point();
        Point pXmax = new Point();
        Point pYmin = new Point();
        Point pYmax = new Point();
        for(int i=0; i<elements.size(); ++i) {
            Point center = elements.get(i);

            if(center.x < cXmin)
            {
                minX = i;
                cXmin = center.x;
                pXmin.x = center.x;
                pXmin.y = center.y;
            }
            if(center.x > cXmax)
            {
                maxX = i;
                cXmax = center.x;
                pXmax.x = center.x;
                pXmax.y = center.y;
            }
            if(center.y > cYmax)
            {
                maxY = i;
                cYmax = center.y;
                pYmax.x = center.x;
                pYmax.y = center.y;
            }
            if(center.y < cYmin)
            {
                minY = i;
                cYmin = center.y;
                pYmin.x = center.x;
                pYmin.y = center.y;
            }

        }
        /*System.out.println(pYmin+"\n"+pXmax+"\n"+pYmax+"\n"+pXmin);
        Mat slaboMi = new Mat(1024, 768, CvType.CV_8UC3);
        Imgproc.drawMarker(slaboMi, pYmin, Scalar.all(255));
        Imgproc.drawMarker(slaboMi, pXmax, new Scalar(255, 0, 0));
        Imgproc.drawMarker(slaboMi, pYmax, new Scalar(0,255,0));
        Imgproc.drawMarker(slaboMi, pXmin, new Scalar(0,0,255));
        FerbTastic.saveFile(slaboMi, "test/"+FerbTastic.counter+"test.jpg");*/
        top.x = pYmin.x;
        top.y = pYmin.y;
        right.x = pXmax.x;
        right.y = pXmax.y;
        bottom.x = pYmax.x;
        bottom.y = pYmax.y;
        left.x = pXmin.x;
        left.y = pXmin.y;
    }

    private Point drawLines(List<MatOfPoint> a, List<MatOfPoint> b, List<MatOfPoint> c, List<MatOfPoint> d, int rows, int cols) {
        Mat mask = new Mat(rows, cols, CvType.CV_8UC1);
        mask.setTo(Scalar.all(0));
        Mat mask2 = new Mat(rows, cols, CvType.CV_8UC1);
        mask2.setTo(Scalar.all(0));
        List<List<MatOfPoint>> list = new ArrayList<List<MatOfPoint>>();
        MatOfPoint points = new MatOfPoint();
        MatOfFloat line = new MatOfFloat();
        points.fromList(Compose.computeCenter(a));
        Imgproc.fitLine(points, line, Imgproc.DIST_HUBER, 0, 1, 1);
        //System.out.println("Line:\n" + line.dump());
        Mat vector = line.rowRange(new Range(0, 2)); // Wektor rownolegly do linii
        //System.out.println("Wektor: " + vector.dump());
        list.add(b);
        list.add(c);
        list.add(d);
        Mat line2 = null;
        for(List<MatOfPoint> element: list)
        {
            line2 = new Mat(); // przechowuje linie dla poszczegolnych konturow
            points.fromList(Compose.computeCenter(element));
            Imgproc.fitLine(points, line2, Imgproc.DIST_HUBER, 0, 1, 1);
            Mat vectorToComp = line2.rowRange(new Range(0, 2)); // Wektor rownolegly do linii
            //System.out.println("Normany: " + vector.dump());
            //System.out.println("Wektor do porownania: " + vectorToComp.dump());
            MatOfFloat result = new MatOfFloat();
            Core.gemm(vector, vectorToComp, 1.0, new Mat()/*Mat.zeros(2, 1, CvType.CV_32FC1)*/, 0.0, result, Core.GEMM_1_T);
            //System.out.println("Wynik mnozenia: " + result.dump());
            float absolute = Math.abs(result.toArray()[0]);
            if(absolute >= 0.0f && absolute <= 0.1f)
            {
                break;
            }else
            {
                line2 = null;
            }
        }
        Mat point = line.rowRange(new Range(2, 4));
        Mat point2 = line2.rowRange(new Range(2,4));
        Mat vector2 = line2.rowRange(new Range(0, 2));
        MatOfFloat pt1 = new MatOfFloat();
        MatOfFloat pt2 = new MatOfFloat();
        Core.multiply(vector, Scalar.all(1000), pt1);
        Core.add(point, pt1, pt1);
        Core.multiply(vector, Scalar.all(-1000), pt2);
        Core.add(point, pt2, pt2);
        Point start = new Point(pt1.toList().get(0), pt1.toList().get(1));
        Point end = new Point(pt2.toList().get(0), pt2.toList().get(1));
        //System.out.println("Start: " + start);
        //System.out.println("End: " + end);
        Imgproc.line(mask, start, end, Scalar.all(1), 1);
        Core.add(mask, mask2, mask2);
        //mask = new Mat(rows, cols, CvType.CV_8UC1);
        mask.setTo(Scalar.all(0));

        pt1 = new MatOfFloat();
        pt2 = new MatOfFloat();
        Core.multiply(vector2, Scalar.all(1000), pt1);
        Core.add(point2, pt1, pt1);
        Core.multiply(vector2, Scalar.all(-1000), pt2);
        Core.add(point2, pt2, pt2);
        start = new Point(pt1.toList().get(0), pt1.toList().get(1));
        end = new Point(pt2.toList().get(0), pt2.toList().get(1));
        System.out.println("Start: " + start);
        System.out.println("End: " + end);
        Imgproc.line(mask, start, end, Scalar.all(1), 1);
        Core.add(mask, mask2, mask2);

        Core.MinMaxLocResult result = Core.minMaxLoc(mask2);
        System.out.println("Najwiekszy: " + result.maxVal);
        Imgproc.drawMarker(mask2, result.maxLoc, Scalar.all(255), Imgproc.MARKER_DIAMOND, 10, 1, Imgproc.LINE_4);
        FerbTastic.saveFile(mask2, "startend/"+FerbTastic.counter+"startend.jpg");
        return result.maxLoc;
    }

    void splitGroup(List<MatOfPoint> filtered, List<Integer> indexes, List<MatOfPoint> A,
                    List<MatOfPoint> B, List<MatOfPoint> C, List<MatOfPoint> D)
    {

        for(int i=0; i<indexes.size(); ++i)
        {
            switch(indexes.get(i))
            {
                case 0:
                    A.add(filtered.get(i));
                    break;
                case 1:
                    B.add(filtered.get(i));
                    break;
                case 2:
                    C.add(filtered.get(i));
                    break;
                case 3:
                    D.add(filtered.get(i));
                    break;
            }
        }
    }

    private boolean isHole(List<MatOfPoint2f> polynomials, Point c)
    {
        boolean output = false;
        for(MatOfPoint2f p: polynomials)
        {
            if(Imgproc.pointPolygonTest(p, c, false) == 1.0)
            {
                output = true;
                break;
            }
        }
        return output;
    }


    private void colorGroups(Mat colors, List<MatOfPoint> filtered, List<Integer> groups) {
        List<MatOfPoint> G0 = new ArrayList<MatOfPoint>();
        List<MatOfPoint> G1 = new ArrayList<MatOfPoint>();
        List<MatOfPoint> G2 = new ArrayList<MatOfPoint>();
        List<MatOfPoint> G3 = new ArrayList<MatOfPoint>();
        for(int i=0;i<filtered.size(); ++i)
        {
            int group = groups.get(i);
            if(group == 0)
            {
                G0.add(filtered.get(i));
            }else if(group == 1)
            {
                G1.add(filtered.get(i));

            }else if(group == 2)
            {
                G2.add(filtered.get(i));

            }else if(group == 3)
            {
                G3.add(filtered.get(i));
            }
        }
        Imgproc.drawContours(colors, G0, -1, Scalar.all(255),-1);
        Imgproc.drawContours(colors, G1, -1, new Scalar(255,0,0),-1);
        Imgproc.drawContours(colors, G2, -1, new Scalar(0,255,0),-1);
        Imgproc.drawContours(colors, G3, -1, new Scalar(0,0,255),-1);
        FerbTastic.saveFile(colors, "colors/"+FerbTastic.counter+"colors.jpg");
    }

    private List<Integer> group(List<MatOfPoint> filtered) {
        List<Point> centers = new ArrayList<Point>();
        for(MatOfPoint c: filtered)
        {
            centers.add(Compose.computeCenter(c));
        }

        MatOfPoint2f matCenters = new MatOfPoint2f();
        matCenters.fromList(centers);

        MatOfInt labels = new MatOfInt();
        Core.kmeans(matCenters, 4, labels, new TermCriteria(), 10, Core.KMEANS_RANDOM_CENTERS);
        return labels.toList();
    }

    boolean classifyCont(MatOfPoint contour)
    {
        boolean output = false;
        Moments moments = Imgproc.moments(contour);
        MatOfDouble hu = new MatOfDouble();
        Imgproc.HuMoments(moments, hu);
        for(Double d: hu.toList())
        {
            System.out.println(d);
        }

        return false;
    }
}
