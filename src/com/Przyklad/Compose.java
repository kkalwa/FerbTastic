package com.Przyklad;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Compose {
    public Compose(Strategy myStrategy, Mat colorImage){
        _strategy = myStrategy;
        image = colorImage;
        toGray(image, grayscale);
    }

    /**
     * Funkcja narysuje na podanym obrazie, podana liste elips kolorem bialym
     * @param list - lista elips
     * @param input - obraz na ktorym rysujemy
     */
    public static void drawEllipses(List<RotatedRect> list, Mat input)
    {
        for(RotatedRect e: list)
        {
            Imgproc.ellipse(input, e, Scalar.all(255));
        }
    }

    public double launch(){
        return _strategy.computeScore(this);
    }

    static void toGray(Mat input, Mat output)
    {
        Imgproc.cvtColor(input, output, Imgproc.COLOR_BGR2GRAY);
    }
    RotatedRect simpleEllipse(MatOfPoint points)
    {
        if(points.toList().size() < 5)
        {
            throw new RuntimeException("Zabraklo punktow do dopasowania elipsy");
        }
        return Imgproc.fitEllipse(new MatOfPoint2f(points.toArray()));
    }

    static double euclidDst(Point p1, Point p2)
    {
        return Math.sqrt( Math.pow(p1.x - p2.x,2.0) + Math.pow(p1.y - p2.y,2.0) );
    }

    static List<Double> euclidDist(List<Point> list, Point center)
    {
        List<Double> output = new ArrayList<Double>();

        for(int i=0; i<list.size(); ++i)
        {
            output.add( euclidDst(list.get(i), center) );
        }
        return output;
    }

    void minMax(List<Double> list, double[] min, double[] max)
    {
        min[0] = list.get(0);
        max[0] = min[0];
        for(Double e: list)
        {
            if(e<min[0])
            {
                min[0] = e;
            }
            if(e>max[0])
            {
                max[0] = e;
            }
        }
    }

    static List<Point> analyzePoints(List<Point> bullList, Point center, boolean maximum)
    {
        List<Double> distances = new ArrayList<Double>();
        Point current = null;

        for (int i = 0; i < bullList.size(); ++i) {
            current = bullList.get(i);
            distances.add(euclidDst(center, current));
        }
        double max = Collections.max(distances);
        double min = Collections.min(distances);
        //System.out.println("Taka jest najwieksza odleglosc: "+max+"\tTaka najmniejsza: "+min);
        Scalar kolor = null;
        List<Integer> indexes = new ArrayList<Integer>(); /*Indeksy punktow ktore wyrzucamy z listy*/
        List<Point> promising = new ArrayList<Point>(); /*Punkty ktore wydaja sie nadawac do aproksymacji elipsy*/
        if(maximum == true) { // Jesli maximum jest rowne true to odnosimy odleglosci do najwiekszej odleglosci od srodka, jesli false to do najmniejszej
            for (int i = 0; i < distances.size(); ++i) {
                if (distances.get(i) / max < 0.9 || distances.get(i) / max > 1.1) {
                    kolor = new Scalar(0, 0, 255);
                } else {
                    kolor = new Scalar(0, 255, 0);
                    promising.add(bullList.get(i));
                }

            }
        }else {
            for (int i = 0; i < distances.size(); ++i) {
                if (distances.get(i) / min < 0.9 || distances.get(i) / min > 1.2) {
                    kolor = new Scalar(0, 0, 255);
                } else {
                    kolor = new Scalar(0, 255, 0);
                    promising.add(bullList.get(i));
                }
            }
        }
        //System.out.println("BullList: " + bullList.size());
        //System.out.println("Promising: " + promising.size());
        return promising;
    }
    static void stdOtsu(Mat input, Mat output)
    {
        Imgproc.threshold(input, output, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
    }

    static Mat stdHist(Mat image)
    {
        List<Mat> listHist = new ArrayList<Mat>();
        listHist.add(image);
        MatOfInt costam = new MatOfInt(0);
        Mat histogram = new Mat();
        MatOfInt histSize = new MatOfInt(256);
        MatOfFloat histRange = new MatOfFloat(0, 255);
        Imgproc.calcHist(listHist, costam, new Mat(),histogram, histSize, histRange);
        System.out.println(listHist.size());
        return histogram;
    }
    static Point computeCenter(MatOfPoint contour)
    {
        Rect rect = Imgproc.boundingRect(contour);
        Point center = new Point();
        center.x = (rect.br().x + rect.tl().x) * 0.5;
        center.y = (rect.br().y + rect.tl().y) * 0.5;
        return center;
    }

    static List<Point> computeCenter(List<MatOfPoint> list)
    {
        List<Point> output = new ArrayList<Point>();
        for(MatOfPoint c: list)
        {
            output.add(computeCenter(c));
        }
        return output;
    }
    static int indexOfMin(List<Double> list)
    {
        int output = -1;
        double min = Collections.min(list);
        for(int i=0; i<list.size(); ++i)
        {
            double d = list.get(i);
            if(d == min)
            {
                output = i;
                break;
            }
        }
        return output;
    }

    static int indexOfMax(List<Double> list)
    {
        int output = -1;
        double max = Collections.max(list);
        for(int i=0; i<list.size(); ++i)
        {
            double d = list.get(i);
            if(d == max)
            {
                output = i;
                break;
            }
        }
        return output;
    }

    static MatOfPoint list2Mat(List<Point> points)
    {
        MatOfPoint output = new MatOfPoint();
        output.fromList(points);
        return output;
    }
    static int biggest(List<Double> list)
    {
        int index = 0;
        double max = list.get(0);

        for(int i=0; i<list.size(); ++i)
        {
            if(list.get(i) > max)
            {
                max = list.get(i);
                index = i;
            }
        }
        return index;
    }
    static List<MatOfPoint2f> createPolys(List<MatOfPoint> contours)
    {
        List<MatOfPoint2f> output = new ArrayList<MatOfPoint2f>();
        for(MatOfPoint c: contours)
        {
            MatOfPoint2f current = new MatOfPoint2f();
            Imgproc.approxPolyDP(new MatOfPoint2f(c.toArray()), current, 1, true);
            output.add(current);
        }

        return output;
    }
static Mat extractBrightest(Mat input)
{
    Mat output = new Mat();
    Imgproc.threshold(input, output, 215, 255, Imgproc.THRESH_BINARY);
    return output;
}
   static MatOfPoint findBullsEye(Mat input, Mat otsu, double[] size)
    {
        Mat gray = new Mat();
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        FerbTastic.saveFile(gray, FerbTastic.additional+"_"+FerbTastic.Xv+"x"+FerbTastic.Yv+"_"+FerbTastic.counter+"gray.jpg");

        Imgproc.medianBlur(gray, gray, 15);
        FerbTastic.saveFile(gray, "gray_rozmyty.jpg");
        Mat otsed = new Mat();
        Imgproc.threshold(gray, otsed, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
        otsu = otsed;

        FerbTastic.saveFile(otsed, "otsu.jpg");
        List<MatOfPoint> listaKonturow = new ArrayList<MatOfPoint>();
        Imgproc.findContours(otsed, listaKonturow, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        //System.out.println("Znaleziono tyle konturow: " + listaKonturow.size());
        Imgproc.cvtColor(otsed, otsed, Imgproc.COLOR_GRAY2BGR);
        Imgproc.drawContours(otsed, listaKonturow, -1, new Scalar(0, 0, 255), 5);
        FerbTastic.saveFile(otsed, FerbTastic.counter + "otsu_kontury.jpg");


        Point[] points = new Point[4];
        List<Double> indicators = new ArrayList<Double>();
        double S, R1;
        for(int i=0; i<listaKonturow.size(); ++i)
        {
            MatOfPoint2f converted = new MatOfPoint2f(listaKonturow.get(i).toArray());
            RotatedRect rect = Imgproc.minAreaRect(converted);
            S = Imgproc.contourArea(converted, true);
            R1 = 2 * S / Math.PI;
            indicators.add(R1);
            //System.out.println(i+": "+indicators.get(i));
        }

        int bullseye = biggest(indicators);
        //System.out.println("Ten indeks: " + bullseye);
        Imgproc.drawContours(otsu, listaKonturow, bullseye, new Scalar(0, 255, 0), 10);
        Rect bounds = Imgproc.boundingRect(listaKonturow.get(bullseye));
        //Imgproc.rectangle(otsu, bounds.tl(), bounds.br(), new Scalar(0,255,0), 1);
        size[0] = euclidDst(bounds.tl(), new Point(bounds.br().x, bounds.tl().y));

        FerbTastic.saveFile(otsu, "otsu_kontury_zielone.jpg");
        otsed.copyTo(otsu);
        return listaKonturow.get(bullseye);
    }


    Mat image;
    Mat grayscale = new Mat();
    private Strategy _strategy;

}
