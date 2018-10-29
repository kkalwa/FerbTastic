package com.Przyklad;



import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.opencv.imgproc.Imgproc.moments;

public class FerbTastic {
    static final String sciezkaStartowa = "tarcze/";
    static final String sciezkaKoncowa = "wynik/";
    static final String sciezkaWzorcow = "wzorce/";
    static int counter = 0;
    static{
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    static Mat openFile(String fileName)
    {
        Mat output = Imgcodecs.imread(sciezkaStartowa + fileName);

        if(output.empty())
        {
            return null;
        }else
        {
            return output;
        }
    }
    static void saveFile(Mat content, String filePath)
    {
        Imgcodecs.imwrite(sciezkaKoncowa + filePath, content);
    }


    static int findBiggestContour(List<MatOfPoint> list)
    {
        double area = 0.0, currentArea;
        int winner = -1, i=0;
        for(MatOfPoint e: list)
        {
            currentArea = Imgproc.contourArea(e);
            if(currentArea > area)
            {
                area = currentArea;
                winner = i;
            }
            ++i;
        }
        return winner;
    }
    /*Jak znalezc srodek konturu*/
    static Point findCenter(List<MatOfPoint> list, int index)
    {
        //Moments moments = Imgproc.moments(list.get(index));
        //Point centralPoint = new Point(moments.m10 / moments.m00, moments.m01 / moments.m00);
        MatOfPoint2f points = new MatOfPoint2f(list.get(index).toArray());
        RotatedRect rectangle = Imgproc.minAreaRect(points);

        return rectangle.center;
    }


    static RotatedRect flood2rect(Mat image, Point startPoint)
    {
        assert(image.channels() == 3);
        int forBigger = 0, forLower = 0;
        Mat local = new Mat(), segmented = new Mat();
        image.copyTo(local); // Stworzenie kopii obrazu wejsciowego (trzeba sie chyba bedzie tego potem pozbyc)
        Imgproc.floodFill(local, new Mat(), startPoint, new Scalar(255,0,0),null,
                new Scalar(forLower,forLower,forLower), new Scalar(forBigger, forBigger, forBigger),Imgproc.FLOODFILL_FIXED_RANGE);
        saveFile(local, "local"+counter+".jpg");
        ++counter;
        Core.inRange(local,new Scalar(255,0,0),new Scalar(255,0,0), segmented);
        List<MatOfPoint> list = new ArrayList<MatOfPoint>();
        Imgproc.findContours(segmented, list, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        MatOfPoint2f current = new MatOfPoint2f(list.get(0).toArray());
        return  Imgproc.minAreaRect(current);
    }


    /*Funkcja ktora tylko wpisuje okna na zdjecie*/
    static void drawCircles(Mat input, Mat circles)
    {
        for(int i=0; i<circles.cols(); ++i)
        {
            double[] current = circles.get(0, i);
            for(double e: current)
            {
                Imgproc.circle(input,new Point(current[0], current[1]), (int)current[2], new Scalar(0,0,255),10);
            }
        }
    }

    /* Funkcja bedzie sluzyc lokalizacji srodka tarczy
    * Poki co znajduje najwiekszy kontur metoda otsu i wyznacza jego srodek
    * Nie jest to faktyczny srodek tarczy, ale jst blisko*/
    static Point localizeCenter(Mat input)
    {
        assert(input.channels() == 3); // Wejscie powinno byc kolorowym obrazem
        Mat gray = new Mat(), otsed = new Mat();
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(gray, gray, 11);
        Imgproc.threshold(gray, otsed, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
        saveFile(otsed, "Otsu.jpg");

        List<MatOfPoint> listaKonturow = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(otsed, listaKonturow, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        int indexOfBiggest = findBiggestContour(listaKonturow);
        Point output = findCenter(listaKonturow, indexOfBiggest);

        MatOfPoint2f points = new MatOfPoint2f(listaKonturow.get(indexOfBiggest).toArray());
        RotatedRect rectangle = Imgproc.minAreaRect(points);
        Point[] pointsArray = new Point[4];
        rectangle.points(pointsArray);
        int dupa = 0;
        pointsArray[1].x+= dupa;
        pointsArray[1].y+= dupa;
        pointsArray[3].x-= dupa;
        pointsArray[3].y-= dupa;
        Rect  rectangle2 = new Rect(pointsArray[1], pointsArray[3]);
        Mat cropped = new Mat(input, rectangle2);
        Imgproc.GaussianBlur(cropped, cropped, new Size(11, 11), 2);
        Imgproc.cvtColor(cropped, cropped, Imgproc.COLOR_BGR2GRAY);
        saveFile(cropped, "cropped.jpg");
        Mat memorize = new Mat();
        cropped.copyTo(memorize);
        List<Mat> listHist = new ArrayList<Mat>();
        listHist.add(cropped);
        MatOfInt costam = new MatOfInt(0);
        System.out.println(costam.dump());
        Mat histogram = new Mat();
        MatOfInt histSize = new MatOfInt(255);
        MatOfFloat histRange = new MatOfFloat(0, 255);
        Imgproc.calcHist(listHist, costam, new Mat(),histogram, histSize, histRange);
        //System.out.println(histogram.dump());
        float[] currentPixel = new float[3];
        for(int i=0; i<255; ++i)
        {
            histogram.get(i, 0, currentPixel);
            System.out.println(i + "\t" + (int)currentPixel[0]);
        }
        Imgproc.threshold(cropped, otsed, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
        saveFile(otsed, "Otsu_cropped.jpg");
        Imgproc.threshold(cropped, cropped, 7,255, Imgproc.THRESH_BINARY_INV);
        saveFile(cropped, "cropped_simple.jpg");
        Imgproc.morphologyEx(cropped, cropped, Imgproc.MORPH_CLOSE,Mat.ones(new Size(50,50), CvType.CV_8UC1));
        saveFile(cropped, "cropped_opened.jpg");
        Mat nowy = new Mat();
        Core.add(cropped, memorize, nowy);
        saveFile(nowy, "Wasacz.jpg");
        Imgproc.adaptiveThreshold(nowy, nowy, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 501, 2);
        saveFile(nowy, "Wasacz_Gauss.jpg");
        return output;
    }
    /*Funkcja powinna wykrywac elipsy na kolorowym obrazie i je rysowac.
    * Wykorzystuje segmentacje metoda Gaussa i algorytm flood fill*/
    static void colorGauss(Mat input)
    {
        Mat gray = new Mat(), gaussian = new Mat(), closed = new Mat(), opened = new Mat(), aux = new Mat();
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.adaptiveThreshold(gray, gaussian, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 501, 2);

        Imgproc.morphologyEx(gaussian, closed, Imgproc.MORPH_CLOSE,Mat.ones(new Size(10,10), CvType.CV_8UC1));
        Imgproc.cvtColor(gaussian, gaussian, Imgproc.COLOR_GRAY2BGR);
        Imgproc.cvtColor(closed, closed, Imgproc.COLOR_GRAY2BGR);
        Point center = localizeCenter(input);
        Imgproc.circle(input, center, 5, new Scalar(0,0,255), 5); /*WARNING!!!!!!*/
        RotatedRect rect = flood2rect(gaussian, center);
        Imgproc.ellipse(input, rect, new Scalar(0,0,255), 1);
        Point realCenter = rect.center;
        Imgproc.circle(input, realCenter, 5, new Scalar(0,0,255), 5); /*WARNING!!!!!!*/
        RotatedRect centerRect = flood2rect(gaussian, realCenter);
        Imgproc.ellipse(input, centerRect, new Scalar(0,0,255), 5);
        Point[] points = new Point[4];

        for(int i=0;i<3;++i)
        {
            rect.points(points);
            points[1].x +=25;
            points[1].y +=25;
            rect = flood2rect(closed, points[1]);
            Imgproc.ellipse(input, rect, new Scalar(0,0,255), 5);
            //Imgproc.circle(input, points[1], 5, new Scalar(0,0,255), 5);
            //rect.points(points);
            //Imgproc.rectangle(input, points[1], points[3], new Scalar(0,0,255), 5);
        }
        Imgproc.morphologyEx(gaussian, opened, Imgproc.MORPH_OPEN,Mat.ones(new Size(10,10), CvType.CV_8UC1));
        for(int i=0; i<0; ++i)
        {
            rect.angle = 0.0;
            rect.points(points);
            center.x = points[1].x -30;
            center.y = points[1].y + rect.size.height/2;
            rect = flood2rect(opened, center);
            Imgproc.ellipse(input, rect, new Scalar(0,0,255), 5);
            Imgproc.circle(input, center, 5, new Scalar(0,0,255), 5);
        }
        saveFile(opened, "Otwarty.jpg");
        saveFile(gaussian, "Gauss.jpg");
        saveFile(input, "elipsy.jpg");

        Mat simpleThresh = new Mat();

        Imgproc.threshold(gray, simpleThresh, 5, 500, Imgproc.THRESH_BINARY_INV);
        Imgproc.morphologyEx(simpleThresh, simpleThresh, Imgproc.MORPH_CLOSE,Mat.ones(new Size(10,10), CvType.CV_8UC1));
        saveFile(simpleThresh, "Simple.jpg");
    }

    /*Funkcja ma za zadanie przyciecie zdjecia tak aby zostalo jak najmniej z otoczenia tarczy*/
static void cropImage(Mat input,Mat output)
{
    Mat gray = new Mat(), otsed = new Mat();
    Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
    Imgproc.medianBlur(gray, gray, 11);
    Imgproc.threshold(gray, otsed, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
    List<MatOfPoint> listaKonturow = new ArrayList<MatOfPoint>();
    Imgproc.findContours(otsed, listaKonturow, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
    int indexOfBiggest = findBiggestContour(listaKonturow);
    MatOfPoint2f points = new MatOfPoint2f(listaKonturow.get(indexOfBiggest).toArray());
    RotatedRect rectangle = Imgproc.minAreaRect(points);
    Point[] pointsArray = new Point[4];
    rectangle.points(pointsArray);


    saveFile(otsed, "OTSU.jpg");

    // Ponizej nalezy sprobowac poprawic przyciecie tak aby nie bylo prostokatow po bokach i zeby wszystko bylo jak najblizej pierwszej obreczy
    int MAX_LOOP = (int) otsed.cols()/2;
    byte[] pixelL = new byte[3];
    byte[] pixelR = new byte[3];
    byte[] pixelU = new byte[3];
    byte[] pixelD = new byte[3]; // Piksele dla prawej, lewej strony i dla gory i dolu
    int centerX = MAX_LOOP; // x w srodku jest akurat rowny MAX_LOOP
    int centerY = (int)otsed.size().height/2; // y w srodku
    boolean  left = true, right = true, down = true, up = true;
    for(int i=0; i<MAX_LOOP; ++i )
    {
        if(left )
            otsed.get(centerY, i, pixelL);
        if(right)
            otsed.get(centerY, otsed.cols()-i, pixelR);
        if(up)
            otsed.get(i, centerX, pixelU);
        if(down)
            otsed.get(otsed.rows() - i, centerX, pixelD);
        /*System.out.println("L: " + pixelL[0]);
        System.out.println("L: " + pixelR[0]);
        System.out.println("L: " + pixelU[0]);
        System.out.println("L: " + pixelD[0]);*/
        if(pixelL[0] != 0 && left)
        {
            pointsArray[1].x = i;
            //System.out.println("L: " + pixelL[0] + "\t" + i);
            left = false;
        }

        if(pixelR[0] != 0 && right)
        {
            pointsArray[3].x = otsed.cols()-i;
            //System.out.println("R: " + pixelR[0] + "\t" + (otsed.cols()-i));
            right = false;
        }

        if(pixelU[0] != 0 && up)
        {

            pointsArray[1].y = i;
            //System.out.println("U: " + pixelU[0] + "\t" + i);
            up = false;
        }

        if(pixelD[0] != 0 && down)
        {
            pointsArray[3].y = otsed.rows() - i;
           // System.out.println("D: " + pixelU[0] + "\t" + (otsed.rows() - i));
            down = false;
        }

        if( !(left || right || up || down)) /*Jak wszyscy sa false to mozna konczyc*/
            break;
    }
    /*Ponizej jeszcze dodatkowe przyciecie*/
    int offset = 35;
    pointsArray[1].x+= offset;
    pointsArray[1].y+= offset;
    pointsArray[3].x-= offset;
    pointsArray[3].y-= offset;
    Rect convertedRect = new Rect(pointsArray[1], pointsArray[3]);
    Mat cropped = new Mat(input, convertedRect);
    cropped.copyTo(output);
}

/*Funkcja znajdzie przestrzeliny*/
static void extractBulletHoles(Mat input, Mat output)
{
    Mat gray = new Mat();
    Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
    Imgproc.medianBlur(gray, gray, 11);
    List<Mat> listHist = new ArrayList<Mat>();
    listHist.add(gray);
    MatOfInt costam = new MatOfInt(0);
    System.out.println(costam.dump());
    Mat histogram = new Mat();
    MatOfInt histSize = new MatOfInt(255);
    MatOfFloat histRange = new MatOfFloat(0, 255);
    Imgproc.calcHist(listHist, costam, new Mat(),histogram, histSize, histRange);

    float[] currentPixel = new float[3];
    for(int i=0; i<255; ++i)
    {
        histogram.get(i, 0, currentPixel);
        System.out.println(i + "\t" + (int)currentPixel[0]);
    }

    Mat thresholded = new Mat();
    Imgproc.threshold(gray, thresholded, 7, 255, Imgproc.THRESH_BINARY_INV);
    Imgproc.morphologyEx(thresholded, thresholded, Imgproc.MORPH_CLOSE,Mat.ones(new Size(50,50), CvType.CV_8UC1));
    thresholded.copyTo(output);
    saveFile(thresholded, "aktualny_dziury.jpg");
}
/*Przeksztalca obraz tak aby kazdy piksel przyjal wartosc najwiekszej odleglosci*/
static void maxDistance(Mat input, Mat output)
{
    assert(input.channels() == 1); // Wejscie powinno byc w skali szarosci
    int rows = input.rows(), cols = input.cols();
    Mat aux = new Mat(input.rows(), input.cols(), CvType.CV_32SC1); // Obiekt bedzie przechowywal wartosci roznic a potem trzeba go bedzie skopiowac do output
    Mat newInput = new Mat(); // Trzeba zdefiniowac nowa macierz input - wejsciowa ma 8 bitow, a powinna miec 32
    input.convertTo(newInput, CvType.CV_32SC1); // tutaj zachodzi konwersja

    Mat neighbour = null; // Otoczenie punktu
    Mat clones = new Mat(3,3, CvType.CV_32SC1); // tablica przechowa 9 kopii przetwarzanego piksela
    Mat results = new Mat(3,3, CvType.CV_32SC1); // wyniki pochodzace z funkcji absdiff()
    int[] pixel = new int[1]; // w tej tablicy przechowywane sa skladowe srodkowego piksela


    for(int i=1; i<rows-1; ++i)
    {
        for(int j=1; j<cols-1; ++j)
        {
            neighbour = newInput.submat(i-1,i+2,j-1,j+2);
            neighbour.get(1,1, pixel);
            clones.setTo(new Scalar(pixel[0], pixel[0], pixel[0]));
            Core.absdiff(neighbour, clones, results);
            Core.MinMaxLocResult minMax = Core.minMaxLoc(results);
            int[] sendIt = {(int)minMax.maxVal};
            aux.put(i, j, sendIt);
        }
    }
    aux.copyTo(output);
    output.convertTo(output, CvType.CV_8UC1);
}
/*Funkcja sprobuje znalezc centralna czesc tarczy (czarne kolo na bialym tle)*/
static MatOfPoint findBullsEye(Mat input, Mat otsu)
{
    Mat gray = new Mat();
    Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
    saveFile(gray, "gray.jpg");

    Imgproc.medianBlur(gray, gray, 15);
    saveFile(gray, "gray_rozmyty.jpg");
    Mat otsed = new Mat();
    Imgproc.threshold(gray, otsed, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

    saveFile(otsed, "otsu.jpg");
    List<MatOfPoint> listaKonturow = new ArrayList<MatOfPoint>();
    Imgproc.findContours(otsed, listaKonturow, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

    //System.out.println("Znaleziono tyle konturow: " + listaKonturow.size());
    Imgproc.cvtColor(otsed, otsed, Imgproc.COLOR_GRAY2BGR);
    Imgproc.drawContours(otsed, listaKonturow, -1, new Scalar(0, 0, 255), 5);
    saveFile(otsed, "otsu_kontury.jpg");


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
    Imgproc.drawContours(otsed, listaKonturow, bullseye, new Scalar(0, 255, 0), 10);
    saveFile(otsed, "otsu_kontury_zielone.jpg");
    otsed.copyTo(otsu);
    return listaKonturow.get(bullseye);
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

static MatOfPoint iterate(MatOfPoint2f points) {

    MatOfPoint convertedP = new MatOfPoint(points.toArray());
    List<Point> list = convertedP.toList();
    Point current = list.get(0);
    Point next = null;
    double dx = 0.0, dy = 0.0;
    List<Point> outputList = new ArrayList<Point>();
    boolean bX = true, bY = true;
    for (int i = 1; i < list.size() - 1; ++i) {
        next = list.get(i);
        dx = next.x - current.x;
        dy = next.y - current.y;

        if(dx>=0) // x sie zwieksza
        {
            if(dy>=0) // x i y sie zwieksza
            {
                if(bX == false || bY == false)
                {
                    outputList.add(new Point(current.x, current.y));
                }
                if(bX == false)
                    bX = true;
                if(bY == false)
                    bY = true;

            }else  // x sie zwieksza, y zmniejsza
            {
                if(bX == false || bY == true)
                {
                    outputList.add(new Point(current.x, current.y));
                }
                if(bX == false)
                    bX = true;
                if(bY == true)
                    bY = false;
            }
        }else // x sie zmniejsza
        {
            if(dy>=0) // x sie zmniejsza, y sie zwieksza
            {
                if(bX == true || bY == false)
                {
                    outputList.add(new Point(current.x, current.y));
                }
                if(bX == true)
                    bX = false;
                if(bY == false)
                    bY = true;
            }else  // x i y sie zmniejszaja
            {
                if(bX == true || bY == true)
                {
                    outputList.add(new Point(current.x, current.y));
                }
                if(bX == true)
                    bX = false;
                if(bY == true)
                    bY = false;
            }
        }
        current = next;
    }
    MatOfPoint output = new MatOfPoint();
    output.fromList(outputList);
    return output;
}

static Mat fitEllipse(MatOfPoint points)
{
    System.out.println("Vector points: " + points);
    Mat x = new Mat(points.rows(), 1, CvType.CV_64FC1);
    Mat y = new Mat(points.rows(), 1, x.depth());
    Mat x2 = new Mat(points.rows(), 1, CvType.CV_64FC1);
    Mat y2 = new Mat(points.rows(), 1, x.depth());
    Mat xy = new Mat(points.rows(), 1, CvType.CV_64FC1);
    Core.extractChannel(points, x, 0);
    Core.extractChannel(points, y, 1);
    x.convertTo(x, CvType.CV_64FC1);
    y.convertTo(y, CvType.CV_64FC1);

    Core.pow(x, 2.0, x2);
    Core.pow(y, 2.0, y2);
    Core.multiply(x,y,xy); /*Mnozenie dwoch wektorow*/
    Mat D1 = new Mat(x.rows(), 3, x.depth());
    Mat D2 = new Mat(x.rows(), 3, x.depth()); /*Macierze D1 i D2 beda przechowywac kwadraty wektorow x i y*/
    System.out.println("Vector x: " + x);
    System.out.println("Vector y: " + y);
    System.out.println("Vector xy: " + xy);
    System.out.println("Matrix D: " + D1);
    double[] value = new double[1];
    for(int i=0; i<D1.rows(); ++i) /*Nie da sie chyba inaczej zlozyc dwoch wektorow w jedna macierz.*/
    {
        x2.get(i,0,value); // Pobranie x do kwadratu
        D1.put(i,0,value); // Wstawienie x do kwadratu
        x.get(i,0,value);  // Pobranie x
        D2.put(i,0,value); // Wstawienie x
        xy.get(i,0,value); // Pobranie iloczynu xy
        D1.put(i,1,value); // Wstawienie iloczynu xy
        y2.get(i,0,value);  // Pobranie y do kwadratu
        D1.put(i,2,value); // Wstawienie y ddo kwadratu
        y.get(i, 0, value); // Pobranie wartosci y
        D2.put(i,1,value); // Wstawienie y
        value[0] = 1;
        D2.put(i,2,value); // Wstawienie wartosci 'jeden' na koniec
    }
    System.out.println("Matrix D1: " + D1);
    System.out.println("Matrix D2: " + D2);
    x = null;
    y = null;
    x2 = null;
    y2 = null;
    xy = null; /*Zwolnienie zasobow*/

    Mat S1 = new Mat(), S2 = new Mat(), S3 = new Mat(), T = new Mat(), M = new Mat();
    System.out.println(D1.cols());

    Core.gemm(D1,D1,1.0,Mat.zeros(D1.size(), D1.depth()),0.0,S1, Core.GEMM_1_T);
    Core.gemm(D1,D2,1.0,Mat.zeros(D2.size(), D2.depth()),0.0,S2, Core.GEMM_1_T);
    Core.gemm(D2,D2,1.0,Mat.zeros(D2.size(), D2.depth()),0.0,S3, Core.GEMM_1_T);
    System.out.println(S3.dump());
    Core.invert(S3,S3);
    System.out.println(S3.dump());
    Core.multiply(S3, Scalar.all(-1.0), S3);
    System.out.println(S3.dump());
    Core.gemm(S3,S2,1.0, Mat.zeros(S3.rows(), S2.cols(),S2.depth()), 0.0, T, Core.GEMM_2_T);
    Core.gemm(S2, T, 1.0, S1, 1.0, M);


    S2 = null;
    S3 = null; /*Zwolnienie zasobow*/
    Mat col1 = M.col(0), col2 = M.col(1), col3 = M.col(2); // Rozbicie na posczegolne kolumny
    Core.multiply(col2, Scalar.all(-1.0), col2);

    for(int i=0; i<S1.rows(); ++i)
    {
        col1.get(i,0,value);
        value[0] /= 2.0;
        M.put(i,2,value);

        col2.get(i,0,value);
        M.put(i,1,value);

        col1.get(i,0,value);
        value[0] /= 2.0;
        M.put(i,0,value);
    }
    col1 = null;
    col2 = null;
    col3 = null;
    Mat evalues = new Mat(), evectors = new Mat();
    Core.eigen(M, evalues, evectors);
    System.out.println("Eigenvalues: " + evalues);
    System.out.println("Eigenvectors: " + evectors);
    System.out.println("Eigenvectors: " + evectors.dump());
    Mat cond = new Mat();
    Mat aux = new Mat();
    Core.multiply(evectors.col(0), evectors.col(2), cond);
    Core.multiply(cond, Scalar.all(4.0), cond);
    Core.pow(evectors.col(1),2.0, aux);
    Core.subtract(cond, aux, cond);
    System.out.println("Cond: " + cond.dump());
    int row = -1;
    for(int i=0; i<cond.rows(); ++i)
    {
        cond.get(i, 0, value);
        if(value[0] > 0)
        {
            row = i;
            break;
        }
    }
    Mat a1 = evectors.col(row);
    System.out.println("a1: " + a1.dump());
    System.out.println("T: " + T.dump());
    Core.gemm(T, a1, 1.0, Mat.zeros(3,1,T.depth()),0.0, T);
    System.out.println("T: " + T.dump());
    Mat a = new Mat(6,1,T.depth());
    for(int i=0; i<3; ++i)
    {
        a1.get(i,0, value);
        a.put(i,0,value);
        T.get(i,0, value);
        a.put(i+3,0,value);
    }
    System.out.println("a: " + a.dump());
    return a;
}

static void ellipseApp(Mat image, MatOfPoint points)
{
    Random generator = new Random();
    List<Point> list = points.toList();
    int streamsize = 20 ;
    if(streamsize > list.size())
    {
        streamsize = list.size();
    }
    IntStream ints = generator.ints(streamsize, 0, list.size());
    int[] array = ints.toArray();
    List<Point> outputList = new ArrayList<Point>();

    for(int i=0; i < array.length; ++i)
    {
        //System.out.println(array[i]);
        outputList.add(list.get(array[i]));
    }
    MatOfPoint2f outputPoints = new MatOfPoint2f();
    outputPoints.fromList(outputList);
    RotatedRect rect = Imgproc.fitEllipse(outputPoints);
    Imgproc.ellipse(image, rect, new Scalar(0,0,255), 5);
    saveFile(image, "Ellipse.jpg");
}
static RotatedRect simpleEllipse(MatOfPoint points)
{
    return Imgproc.fitEllipse(new MatOfPoint2f(points.toArray()));
}

static List<RotatedRect> segments(Mat image, MatOfPoint bullseye, Point center)
{
    List<RotatedRect> allRect = new ArrayList<RotatedRect>();
    Mat gray = new Mat();
    toGray(image, gray);
    //approxEllipse(bullseye, center, image);
    //ellipseApp(image, bullseye);
    allRect.add(simpleEllipse(bullseye));
    MatOfDouble mean = new MatOfDouble(), stdDev = new MatOfDouble(); /*Momenty wykorzystywane potem*/
    Mat mask = new Mat(gray.rows(), gray.cols(), gray.depth());
    double kappa = 0.0;
    List<Point> list = bullseye.toList();
    Mat sobel = new Mat();
    Imgproc.Sobel(gray, sobel, CvType.CV_8UC1, 1, 1); // Obliczenie pochodnej metoda sobela
    Point test = new Point();
    double L = 0.0;
    Rect rect = Imgproc.boundingRect(bullseye);
    double specifiedL = rect.width/6.0; // Tutaj okreslic dlugosc jaka nas interesuje
    double alfa = 0.3;                  // Alfa jest uzywane przy obliczaniu kappa
    Point next = new Point();
    List<Point> ring = null;
    List<Double> distances = new ArrayList<Double>();
    Scalar[] colors = {Scalar.all(0),new Scalar(255,0,0), new Scalar(0,255,0), new Scalar(0,0,255)};
    boolean stop = true; /*Okresla kiedy konczymy petle*/
    int a=0; // Licznik do petli
    Point direction = null;

    while(stop == true){
        ring = new ArrayList<Point>();
        counter = 0;
        for (int i = 0; i < list.size(); ++i) {
            Point current = list.get(i);
            direction = new Point(center.x - current.x, center.y - current.y); // To wlasciwie nie punkt, bardziej wektor wyznaczajacy zwrot
            L = Math.sqrt( Math.pow(direction.x, 2) + Math.pow(direction.y, 2)); /*Odleglosc miedzy punktem na obwodzie a punktem srodkowym*/
            if(L <= specifiedL/2.0) {
                stop = false;
                break;
            }
            current.x += (direction.x / L)*specifiedL/2.0;
            current.y += (direction.y / L)*specifiedL/2.0; /* Zmiana wspolrzednych punktu na pierscieniu (chodzi o to by nie brac pod uwage tego punktu, bo zazwyczaj jest ekstremum)*/
            test.x = current.x + (direction.x / L) * specifiedL/2.0;
            test.y = current.y + (direction.y / L) * specifiedL/2.0; /* Punkty current oraz test wyznaczaja odcinek z ktorego piksele bierzemy pod uwage*/
            Imgproc.line(mask, test, current, Scalar.all(255));
            Core.MinMaxLocResult result = Core.minMaxLoc(gray, mask); /* Z linii odczytujemy najwieksza i najmniejsza wartosc*/
            Core.meanStdDev(gray, mean, stdDev, mask); /*Sciagamy z linii srednia i odchylenie standardowe*/
            kappa = (mean.toArray()[0] - result.minVal) / (result.maxVal - result.minVal); /* Wskaznik pomaga okreslic jaki typ pierscienia mamy*/
            //System.out.println(i + ". Srednia: " + mean.toArray()[0] + "\tOdchylenie: " + stdDev.toArray()[0] + "\tKappa: " + kappa);
            if (kappa > 1 - alfa) { /* W zaleznosci od kappa punkt na wewnetrznym pierscieniu przyjmuje odpowiednia postac*/
                next.x = result.minLoc.x;
                next.y = result.minLoc.y;
            } else if (kappa < alfa) {
                next.x = result.maxLoc.x;
                next.y = result.maxLoc.y;
            } else {
                Core.MinMaxLocResult diff = Core.minMaxLoc(sobel, mask); // Jezeli kappa jest bliska 0.5 to bierze sie punkt o najwiekszej pochodnej
                next.x = diff.maxLoc.x;
                next.y = diff.maxLoc.y;
            }
            Imgproc.circle(image, next, 10, colors[a], 5); /*!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!*/
            Imgproc.line(image, current, test, colors[a]);
            //System.out.println((counter++ +1) +" odl: " + Math.sqrt( Math.pow(next.x - current.x, 2) + Math.pow(next.y - current.y, 2)));
            distances.add(Math.sqrt( Math.pow(next.x - current.x, 2) + Math.pow(next.y - current.y, 2)));
            ring.add(new Point(next.x, next.y));
            mask.setTo(Scalar.all(0));
        }
        if(distances.size() > 0) {
            double avg = distances.stream().mapToDouble(e -> e).average().getAsDouble(); /*Wartosc srednia odleglosci od srodka*/
            //System.out.println("Srednia: " + avg);
            for (int i = distances.size() - 1; i >= 0; --i) {
                double current = distances.get(i);
                if (current > 1.3 * avg || current < 0.5 * avg) {
                    //Point point = ring.get(i);
                    //point.x = center.x + (direction.x / L) * avg;
                    //point.y = center.y + (direction.y / L) * avg;
                    ring.remove(i);
                }
            }
            distances.clear();
        }

        if(stop == true) {
            MatOfPoint points = new MatOfPoint();
            points.fromList(ring);
            //approxEllipse(points, center, image);
            //ellipseApp(image, points);
            allRect.add(simpleEllipse(points));
        }
        list = ring;
        ++a;
        System.out.println(a);

    }

    stop = true;
    list = bullseye.toList();
   // while(stop == true){
    for(int b=0; b<6; ++b) {
        ring = new ArrayList<Point>();
        for (int i = 0; i < list.size(); ++i) {
            Point current = list.get(i);
            direction = new Point(center.x - current.x, center.y - current.y); // To wlasciwie nie punkt, bardziej wektor wyznaczajacy zwrot
            L = Math.sqrt(Math.pow(direction.x, 2) + Math.pow(direction.y, 2)); /*Odleglosc miedzy punktem na obwodzie a punktem srodkowym*/
            if (L <= specifiedL / 2.0) {
                stop = false;
                break;
            }
            current.x -= (direction.x / L) * specifiedL / 2.0;
            current.y -= (direction.y / L) * specifiedL / 2.0; /* Zmiana wspolrzednych punktu na pierscieniu (chodzi o to by nie brac pod uwage tego punktu, bo zazwyczaj jest ekstremum)*/
            test.x = current.x - (direction.x / L) * specifiedL / 2.0;
            test.y = current.y - (direction.y / L) * specifiedL / 2.0; /* Punkty current oraz test wyznaczaja odcinek z ktorego piksele bierzemy pod uwage*/
            Imgproc.line(mask, test, current, Scalar.all(255));
            Core.MinMaxLocResult result = Core.minMaxLoc(gray, mask); /* Z linii odczytujemy najwieksza i najmniejsza wartosc*/
            Core.meanStdDev(gray, mean, stdDev, mask); /*Sciagamy z linii srednia i odchylenie standardowe*/
            kappa = (mean.toArray()[0] - result.minVal) / (result.maxVal - result.minVal); /* Wskaznik pomaga okreslic jaki typ pierscienia mamy*/
            //System.out.println(i + ". Srednia: " + mean.toArray()[0] + "\tOdchylenie: " + stdDev.toArray()[0] + "\tKappa: " + kappa);
            if (kappa > 1 - alfa) { /* W zaleznosci od kappa punkt na wewnetrznym pierscieniu przyjmuje odpowiednia postac*/
                next.x = result.minLoc.x;
                next.y = result.minLoc.y;
            } else if (kappa < alfa) {
                next.x = result.maxLoc.x;
                next.y = result.maxLoc.y;
            } else {

                Core.MinMaxLocResult diff = Core.minMaxLoc(sobel, mask);
                next.x = diff.maxLoc.x;
                next.y = diff.maxLoc.y;
            }
            //Imgproc.circle(image, next, 10, colors[2], 5);
            //Imgproc.line(image, current, test, colors[2]);
            ring.add(new Point(next.x, next.y));
            mask.setTo(Scalar.all(0));
        }
        if (stop == true) {
            MatOfPoint points = new MatOfPoint();
            points.fromList(ring);
            //approxEllipse(bullseye, center, image);
            //ellipseApp(image, points);
            allRect.add(simpleEllipse( points));

        }
        list = ring;
        ++a;
        System.out.println(b);
    }
    //}

    saveFile(image, "punkty.jpg");

    return allRect;
}
static Mat line2Mat(Mat image,Point start, Point end)
{
    Mat output = new Mat(image.rows(), image.cols(), image.depth());
    Imgproc.line(output, start, end, Scalar.all(255));
    Core.inRange(output, Scalar.all(255), Scalar.all(255),output);
    List<MatOfPoint> list = new ArrayList<MatOfPoint>();
    Imgproc.findContours(output, list, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
    MatOfPoint points = list.get(0);
    List<Point> pointsList = points.toList();
    output = null;
    output = new Mat(1,pointsList.size(), CvType.CV_32SC3);
    for(int i=0; i<pointsList.size(); ++i)
    {
        Point e = pointsList.get(i);
        int[] pixel = new int[3];
        image.get((int)e.y,(int)e.x, pixel);
        output.put(0,i,pixel);
    }

    return output;
}

static void approxEllipse(MatOfPoint points, Point center, Mat image)
{
    List<Point> list = points.toList();
    List<Double> distances = new ArrayList<Double>();
    double distance = 0.0;
    Point current = null;
    for(int i=0; i<list.size() ; ++i)
    {
        current = list.get(i);
        distance = Math.sqrt( Math.pow(center.x - current.x,2) + Math.pow(center.y - current.y,2) );
        distances.add(distance);
        System.out.println(distance);
    }

   double mean = distances.stream()
            .mapToDouble(e -> e)
            .average()
            .getAsDouble();
    System.out.println("Srednia: " + mean);
    System.out.println("Tyle elementow ma lista przed filtracja: " + list.size());
    List<Point> reduced = list.stream()
            .filter(p -> Math.sqrt( Math.pow(center.x - p.x,2) + Math.pow(center.y - p.y,2) ) < mean)
            .collect(Collectors.toList());

    System.out.println("Tyle elementow ma lista po filtracji: " + reduced.size());
    System.out.println("Tyle elementow ma lista po filtracji (stara lista): " + list.size());

    /*Random generator = new Random();
    int streamsize = 5;
    IntStream ints = generator.ints(streamsize, 0, reduced.size())
            .collect(Collectio);*/
    MatOfPoint2f ellipsePoints = new MatOfPoint2f();
    ellipsePoints.fromList(reduced);
    RotatedRect rect = Imgproc.fitEllipse(ellipsePoints);

//    Imgproc.cvtColor(image, image, Imgproc.COLOR_GRAY2BGR);
  //  image.convertTo(image, CvType.CV_32SC3);
    Imgproc.ellipse(image, rect, new Scalar(0,0,255));
    //int[] red = {0,0,255};

    //reduced.forEach(p -> image.put((int)p.y, (int)p.x, red));
    //saveFile(image, "Strumienie.jpg");
    center.x = rect.center.x;
    center.y = rect.center.y;
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
static void printMat(Mat input)
{
    int rows = input.rows();
    int columns = input.cols();
    int[] value = new int[1];
    int depth = input.depth();
    boolean conversion = false;
    if(depth != CvType.CV_32SC1);
    {
        input.convertTo(input, CvType.CV_32SC1);
        conversion = true;
    }
    for(int i=0; i<rows; ++i)
    {
        for(int j=0; j<columns; ++j)
        {
            input.get(i, j, value);
            System.out.print(value[0] + "\t");
        }
        System.out.println();
    }

    if(conversion == true)
    {
        input.convertTo(input, depth);
    }
}

static void toGray(Mat input, Mat output)
{
    Imgproc.cvtColor(input, output, Imgproc.COLOR_BGR2GRAY);
}


static List<MatOfPoint> basedonMrp(Mat input, MatOfPoint bullseye, Point center)
{
    Mat mask = new Mat(input.rows(), input.cols(), CvType.CV_8UC1); /*Sluzy do wyciecia centralnej czesci tarczy*/
    Mat gray = new Mat();
    toGray(input, gray);
    mask.setTo(Scalar.all(0));
    Imgproc.fillConvexPoly(mask, bullseye, Scalar.all(255));
    saveFile(mask, "maska_bullseye.jpg");
    Mat cropped = new Mat();
    gray.copyTo(cropped, mask);
    saveFile(cropped, "przyciety.jpg");
    Mat otsu = new Mat();
    Imgproc.threshold(cropped, otsu, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
    saveFile(otsu, "cropped_otsu.jpg");
    Mat morphology = new Mat();
    Imgproc.morphologyEx(otsu,morphology, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(11,11)));
    saveFile(morphology, "cropped_otwarty.jpg");
    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    Imgproc.findContours(morphology, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE); // Pobranie listy konturow po morfologii (a wiec po usunieciu niepotrzebnych rzeczy)
    //System.out.println("Liczba konturow: " + contours.size());

    MatOfPoint2f poly = new MatOfPoint2f();
    List<MatOfPoint2f> polys = new ArrayList<MatOfPoint2f>();
    MatOfPoint2f centerPoly = null;
    double retTest = 0.0;
    int i=0, out=-1;
    List<Integer> indexes = new ArrayList<Integer>();
    List<Double> heights = new ArrayList<Double>();
    for(MatOfPoint e: contours)
    {
        Imgproc.approxPolyDP(new MatOfPoint2f(e.toArray()), poly,1,true);

        RotatedRect rect = Imgproc.minAreaRect(new MatOfPoint2f(e.toArray()));
        //System.out.println((i+1) + ". Wysokosc: " + rect.size.height);
        heights.add(rect.size.height);
        ++i;
    }
    double srednia = heights.stream().mapToDouble(e -> e).average().getAsDouble();
    //System.out.println("Srednia wysokosc: " + srednia);
    Imgproc.morphologyEx(morphology, morphology, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(srednia*0.5, srednia*0.5)));
    saveFile(morphology, "nieLubieSiebie.jpg"); // Otwarcie zeby powiekszyc nieco rozmiary przestrzelin i upodobnic je do okregow
    Mat gdzie = new Mat(input.rows(), input.cols(), CvType.CV_8UC3);
    /*for(int e: indexes) {
        Imgproc.drawContours(gdzie, contours, e, Scalar.all(255));
    }
    saveFile(gdzie, "tutaj.jpg");*/ //Te petle mozna chyba usunac ZAPAMIETAC!!!!!!!!!!!!!
    contours.clear();
    Imgproc.findContours(morphology, contours, new Mat(), Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
    i=0;

    for(MatOfPoint e: contours) // W petli nastepuje sprawdzenie wenatrz ktorego konturu znajduje sie punkt centralny. W ten sposob lokalizuje sie biale kolo na srodku
    {
        Imgproc.approxPolyDP(new MatOfPoint2f(e.toArray()), poly,1,true);
        retTest = Imgproc.pointPolygonTest(poly, center, false);
        //System.out.println(retTest);
        if(retTest == 1.0) // Zapamietuje indeksy konturow w ktorych jest center (moze ich byc kilka)
        {
            indexes.add(i);
            centerPoly = poly; // Tutaj jest zachowywana referencja do centralnego kola
        }else
        {
            polys.add(poly); // Jesli center nie nalezy do danego konturu to przechowac do dalszego przetwarzania
        }
    }
    //System.out.println("Centerpoly: " + centerPoly);
    //System.out.println("Tyle przestrzelin: " + polys.size());


    /*Mat pochodna = ScharrDer(gray);

    saveFile(pochodna, "der.jpg");
    Mat derThresh = new Mat();
    stdOtsu(gray, otsu);
    saveFile(otsu, "takwygladaotsu.jpg");
    double avg = Core.mean(gray, otsu).val[0];
    System.out.println("Srednia: " + avg);
    saveFile(mask, "takwygladamaska.jpg");*/
    double avgBull = Core.mean(gray, mask).val[0];
    //System.out.println("Srednia bullseye: " + avgBull);
    //mask.setTo(Scalar.all(0));
    double avgHoles = 0.0;
    //System.out.println("Chwile przed wejsciem do petli: " + polys.size());
    mask.setTo(Scalar.all(0));
    for(MatOfPoint2f e: polys) // Pobranie sredniej jasnosci przestrzelin (znajdujacych sie w obszarze bullseye!)
    {
        Imgproc.fillConvexPoly(mask, new MatOfPoint(e.toArray()), Scalar.all(255));
        avgHoles += Core.mean(gray, mask).val[0];
    }
    avgHoles /= polys.size();
    //System.out.println("Srednia przestrzelin: " + avgHoles);
    Mat simple = new Mat();
    Imgproc.threshold(gray, simple, 0.85*avgHoles, 255, Imgproc.THRESH_BINARY);
    //saveFile(simple, "simple.jpg");
    Imgproc.morphologyEx(simple, simple, Imgproc.MORPH_DILATE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5,5))); // Dylatacja wzmocni przestrzeliny
    saveFile(simple, "simple.jpg");
    contours.clear();
    Imgproc.findContours(simple, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
    //System.out.println("W calym obrazie znaleziono tyle przestrzelin: " + contours.size());
    return contours;
}

static Mat ScharrDer(Mat input)
{
    Mat gray = input;
    Mat scharrX = new Mat(), scharrY = new Mat();
    Imgproc.Scharr(gray, scharrX, CvType.CV_8U, 1, 0);
    saveFile(scharrX, "Pochodna_x.jpg");
    Imgproc.Scharr(gray, scharrY, CvType.CV_8U, 0, 1);
    saveFile(scharrY, "Pochodna_y.jpg");
    Mat combined = new Mat();
    Core.add(scharrX, scharrY, combined);
    saveFile(combined, "ScharrDerivative.jpg");
    return combined;
}
static double euclidDst(Point p1, Point p2)
{
    return Math.sqrt( Math.pow(p1.x - p2.x,2.0) + Math.pow(p1.y - p2.y,2.0) );
}

static void drawPoints(Mat input, List<Point> list)
{
    for(Point e: list)
    {
        Imgproc.circle(input, e, 10, new Scalar(0,0,255), 5);
    }
}
static List<RotatedRect> segments2(Mat image, MatOfPoint bullseye, Point center)
{
    Mat gray = new Mat();
    toGray(image, gray);    /*Szare zdjecie*/

    List<Point> actualPoints = bullseye.toList(); /* Zamiana bullseye na liste punktow*/
    Size bullSize = Imgproc.boundingRect(bullseye).size(); /* Wymiary prostokata wokol bullseye*/

    double L = 0.0;                                      /*Odleglosc od punktu na obwodzie do srodka*/
    double specifiedL = Math.min(bullSize.height, bullSize.width)/6.0; /*Tutaj nalezy okreslic rozmiar obszaru poszukiwan (czyli dlugosc odcinka w ktorym poszukujemy)*/
    double kappa = 0.0; /*Wskaznik okreslajacy jaki typ pierscienia mamy*/
    double alfa = 0.3;  /*Parametr okreslajacy jakie jest tlo*/
    double mean = 0.0; /*Tutaj przechowywac srednia*/

    Mat mask = new Mat(gray.rows(), gray.cols(), CvType.CV_8UC1); /*Maska bedzie okreslac segment z ktorego pobieramy wartosci*/
    //Mat derivative = ScharrDer(gray); /*Obliczenie pochodnej w obu kierunkach metoda Scharra*/
    Mat derivative = new Mat();
    Imgproc.Sobel(gray, derivative, CvType.CV_8UC1, 1, 1);
    saveFile(derivative, "SobelDerivative.jpg");

    Core.MinMaxLocResult result = null; /*W obiekcie przechowywane beda wartosci ekstremalne*/
    Point direction = new Point(); /*Kierunek wyznaczajacy linie*/
    Point current = null;  /*Punkt 'wyjmowany' z listy w index-tej iteracji*/
    Point end = new Point(); /*Linia bedzie rysowana miedzy punktem current, a punktem end*/
    Point next = new Point(); /*Punkt powstaly z current po przejsciu blizej center*/
    List<Point> promising = null; /*Punkty ktore wydaja sie dobrymi kandydatami do aproksymacji elipsy*/
    List<Point> ring = new ArrayList<Point>(); /*Beda tu ladowac punkty nazywane next, potem z nich beda wybierane odpowiednie i przechowywane w promising*/
    MatOfPoint2f ell = new MatOfPoint2f();
    RotatedRect ellRect = null;
    List<RotatedRect> allRects = new ArrayList<RotatedRect>(); /*Tutaj przechowywac elipsy i ten obiekt potem sie zwraca*/
    allRects.add(Imgproc.fitEllipse(new MatOfPoint2f(bullseye.toArray()))); /*Pierwsza elipsa reprezentuje bullseye*/

    Scalar kolor = null; /*!!!!!!!!!!!!!!!!!!!!!!!!Uchwyt na kolory USUNAC TO POTEM!!!!!!!!!!!!!!!!!!!!!!!!*/
    int direction2 = 1; /*Drugi kierunek odpowiada za sterowanie kierunku poszukiwan - -1 na zewnatrz albo 1 do wewnatrz*/

    for(int i=0; i<9; ++i) {
        for (int index = 0; index < actualPoints.size(); ++index) // !!!!!ZAPAMIETAC: index - numer punktu
        {
            current = actualPoints.get(index); /*Na poczatku pobieram punkt z listy*/
            direction.x = center.x - current.x;
            direction.y = center.y - current.y;       /*okreslam kierunek center - current*/
            L = euclidDst(center, current);           /*Odleglosc miedzy punktem srodkowym a punktem na obwodzie*/
            current.x += direction2*(direction.x / L) * specifiedL / 1.5;
            current.y += direction2*(direction.y / L) * specifiedL / 1.5; /*Przeniesienie current blizej srodka (WAZNE !!!!!!!!!!!)*/
            end.x = current.x + direction2*(direction.x / L) * specifiedL / 3.0;
            end.y = current.y + direction2*(direction.y / L) * specifiedL / 3.0; /* Punkty current oraz end wyznaczaja odcinek z ktorego piksele bierzemy pod uwage*/

            mask.setTo(Scalar.all(0));          /* maske trzeba ciagle wyzerowywac*/
            Imgproc.line(mask, current, end, Scalar.all(255)); /*Nakreslenie linii na masce, ktora okresla obszar poszukiwan WAZNE!!!!!!!!!!!!!!*/
            //saveFile(mask, "maska.jpg");

            result = Core.minMaxLoc(gray, mask); /* Z linii odczytujemy najwieksza i najmniejsza wartosc*/
            mean = Core.mean(gray, mask).val[0];
            kappa = (mean - result.minVal) / (result.maxVal - result.minVal); /* Wskaznik pomaga okreslic jaki typ pierscienia mamy*/

            if (kappa > 1 - alfa) { /* W zaleznosci od kappa punkt na wewnetrznym pierscieniu przyjmuje odpowiednia postac*/
                next.x = result.minLoc.x;
                next.y = result.minLoc.y;
                kolor = new Scalar(255, 0, 0);
                if(direction2 < 0) {
                    ring.add(new Point(next.x, next.y));
                }
            } else if (kappa < alfa) {
                next.x = result.maxLoc.x;
                next.y = result.maxLoc.y;
                kolor = new Scalar(0, 255, 0);
                if(direction2 > 0) {
                    ring.add(new Point(next.x, next.y));
                }

            } else {
                Core.MinMaxLocResult diff = Core.minMaxLoc(derivative, mask);
                next.x = diff.maxLoc.x;
                next.y = diff.maxLoc.y;
                kolor = new Scalar(0, 0, 255);
                ring.add(new Point(next.x, next.y));
            }

            //Imgproc.line(image, current, end, new Scalar(0, 0, 255));
            //Imgproc.circle(image, next, 10, kolor, 1);       // Wypisanie punktow obliczonych w tej iteracji !!!!!!!!!USUNAC POTEM!!!!!!!!!!!!!!!!!

        }
        /*!!!!!!!!!!!!!!!!ZAPAMIETAC!!!!!!!!!!! - ring na tym etapie moze byc pusty!*/
        if(ring.size() > 0 && ring.size() >= 5) {
            actualPoints = analyzePoints(ring, center);
            //System.out.println("Actual przed wyczyszczeniem ring:" + actualPoints.size());
            //drawPoints(image,actualPoints);
            //saveFile(image, "points.jpg");

            if(actualPoints.size() >= 5) {
                ring.clear();
                ell.fromList(actualPoints);
                ellRect = Imgproc.fitEllipse(ell);
                allRects.add(ellRect);
                Imgproc.ellipse(image, ellRect, new Scalar(0, 0, 255), 1);
                //System.out.println("Actual po wyczyszczeniu ring:" + actualPoints.size());
                //System.out.println("W " + (i+1) +" iteracji zostalo "+actualPoints.size()+" punktow"); /*USUNAC POTEM*/
            }else
            {
                System.out.println("Ring nie byl pusty, ale nie mamy juz 5 punktow, iteracja: " + (i+1));
                ell.fromList(ring);
                ellRect = Imgproc.fitEllipse(ell);
                allRects.add(ellRect);
                ring.clear();
                actualPoints = bullseye.toList();
                direction2 = -1;
                continue;

            }
        }else
        {
            System.out.println("Ring byl pusty, lub mial mniej niz 5 punktow w iteracji: "+ (i+1));
            actualPoints = bullseye.toList();
            direction2 = -1;
            continue;
        }
    }

    saveFile(image, "depresja.jpg");
    return allRects;
}

    static List<RotatedRect> segments3(Mat image, MatOfPoint bullseye, Point center) {
        Mat gray = new Mat();
        toGray(image, gray);    /*Szare zdjecie*/

        List<Point> actualPoints = bullseye.toList(); /* Zamiana bullseye na liste punktow*/
        Size bullSize = Imgproc.boundingRect(bullseye).size(); /* Wymiary prostokata wokol bullseye*/

        double L = 0.0;                                      /*Odleglosc od punktu na obwodzie do srodka*/
        double specifiedL = Math.min(bullSize.height, bullSize.width) / 6.0; /*Tutaj nalezy okreslic rozmiar obszaru poszukiwan (czyli dlugosc odcinka w ktorym poszukujemy)*/
        double kappa = 0.0; /*Wskaznik okreslajacy jaki typ pierscienia mamy*/
        double alfa = 0.3;  /*Parametr okreslajacy ktora jakie jest tlo*/
        double mean = 0.0; /*Tutaj przechowywac srednia*/

        Mat mask = new Mat(gray.rows(), gray.cols(), CvType.CV_8UC1); /*Maska bedzie okreslac segment z ktorego pobieramy wartosci*/
        Mat derivative = ScharrDer(gray); /*Obliczenie pochodnej w obu kierunkach metoda Scharra*/

        Core.MinMaxLocResult result = null; /*W obiekcie przechowywane beda wartosci ekstremalne*/
        Point direction = new Point(); /*Kierunek wyznaczajacy linie*/
        Point current = null;  /*Punkt 'wyjmowany' z listy w index-tej iteracji*/
        Point end = new Point(); /*Linia bedzie rysowana miedzy punktem current, a punktem end*/
        Point next = new Point(); /*Punkt powstaly z current po przejsciu blizej center*/
        List<Point> promising = null; /*Punkty ktore wydaja sie dobrymi kandydatami do aproksymacji elipsy*/
        List<Point> ring = new ArrayList<Point>(); /*Beda tu ladowac punkty nazywane next, potem z nich beda wybierane odpowiednie i przechowywane w promising*/
        MatOfPoint2f ell = new MatOfPoint2f();
        RotatedRect ellRect = null;
        List<RotatedRect> allRects = new ArrayList<RotatedRect>(); /*Tutaj przechowywac elipsy i ten obiekt potem sie zwraca*/
        allRects.add(Imgproc.fitEllipse(new MatOfPoint2f(bullseye.toArray()))); /*Pierwsza elipsa reprezentuje bullseye*/

        Scalar kolor = null; /*!!!!!!!!!!!!!!!!!!!!!!!!Uchwyt na kolory USUNAC TO POTEM!!!!!!!!!!!!!!!!!!!!!!!!*/
        int direction2 = 1; /*Drugi kierunek odpowiada za sterowanie kierunku poszukiwan - -1 na zewnatrz albo 1 do wewnatrz*/

        int randomCount = 10; // Ile liczb wylosowac
        Random r = new Random();
        int max = actualPoints.size()-1, min = 0;
        int[] random = new int[randomCount];
        for(int i=0; i<randomCount; ++i)
        {
            random[i] = r.nextInt(max-min+1) + min;
        }

        for (int index = 0; index < randomCount; ++index) // !!!!!ZAPAMIETAC: index - numer punktu
        {
            current = actualPoints.get(random[index]); /*Na poczatku pobieram punkt z listy*/
            direction.x = center.x - current.x;
            direction.y = center.y - current.y;       /*okreslam kierunek center - current*/
            L = euclidDst(center, current);           /*Odleglosc miedzy punktem srodkowym a punktem na obwodzie*/
            current.x += direction2 * (direction.x / L) * specifiedL / 1.5;
            current.y += direction2 * (direction.y / L) * specifiedL / 1.5; /*Przeniesienie current blizej srodka (WAZNE !!!!!!!!!!!)*/
            end.x = current.x + direction2 * (direction.x / L) * specifiedL / 3.0;
            end.y = current.y + direction2 * (direction.y / L) * specifiedL / 3.0; /* Punkty current oraz end wyznaczaja odcinek z ktorego piksele bierzemy pod uwage*/

            mask.setTo(Scalar.all(0));          /* maske trzeba ciagle wyzerowywac*/
            Imgproc.line(mask, current, end, Scalar.all(255)); /*Nakreslenie linii na masce, ktora okresla obszar poszukiwan WAZNE!!!!!!!!!!!!!!*/
            //saveFile(mask, "maska.jpg");

            result = Core.minMaxLoc(gray, mask); /* Z linii odczytujemy najwieksza i najmniejsza wartosc*/
            mean = Core.mean(gray, mask).val[0];
            kappa = (mean - result.minVal) / (result.maxVal - result.minVal); /* Wskaznik pomaga okreslic jaki typ pierscienia mamy*/

            if (kappa > 1 - alfa) { /* W zaleznosci od kappa punkt na wewnetrznym pierscieniu przyjmuje odpowiednia postac*/
                next.x = result.minLoc.x;
                next.y = result.minLoc.y;
                kolor = new Scalar(255, 0, 0);
                if (direction2 < 0) {
                    ring.add(new Point(next.x, next.y));
                }
            } else if (kappa < alfa) {
                next.x = result.maxLoc.x;
                next.y = result.maxLoc.y;
                kolor = new Scalar(0, 255, 0);
                if (direction2 > 0) {
                    ring.add(new Point(next.x, next.y));
                }

            } else {
                Core.MinMaxLocResult diff = Core.minMaxLoc(derivative, mask);
                next.x = diff.maxLoc.x;
                next.y = diff.maxLoc.y;
                kolor = new Scalar(0, 0, 255);
            }

            //Imgproc.line(image, current, end, new Scalar(0, 0, 255));
            //Imgproc.circle(image, next, 10, kolor, 5);       // Wypisanie punktow obliczonych w tej iteracji !!!!!!!!!USUNAC POTEM!!!!!!!!!!!!!!!!!

        }
        /*!!!!!!!!!!!!!!!!ZAPAMIETAC!!!!!!!!!!! - ring na tym etapie moze byc pusty!*/
        if (ring.size() > 0 && ring.size() >= 5) {
            promising = analyzePoints(ring, center);
            if(promising.size() < randomCount)
            {
                System.out.println("Za malo punktow: " + promising.size());
                for(int d=0; d <= randomCount - promising.size(); ++d)
                {
                    promising.add(actualPoints.get(r.nextInt(max-min+1) + min));
                }
            }
            System.out.println("Promising: " + promising.size());
            drawPoints(image, promising);
            //System.out.println("Actual przed wyczyszczeniem ring:" + actualPoints.size());
            //drawPoints(image,actualPoints);
            //saveFile(image, "points.jpg");


            saveFile(image, "depresja.jpg");
            return allRects;
        }
        return null;
    }
static List<Point> analyzePoints(List<Point> bullList, Point center)
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

    for(int i=0; i<distances.size(); ++i)
    {
        if(distances.get(i)/max <0.9 || distances.get(i)/max > 1.1)
        {
            kolor = new Scalar(0,0,255);
        }else
        {
            kolor = new Scalar(0,255,0);
            promising.add(bullList.get(i));
        }

    }
    //System.out.println("BullList: " + bullList.size());
    //System.out.println("Promising: " + promising.size());
    return promising;
}

static List<MatOfPoint> findNumbers(Mat image)
{
    Mat gray = new Mat();
    Mat otsu = new Mat();
    toGray(image, gray);
    stdOtsu(gray, otsu);
    saveFile(otsu, "findNumbers.jpg");
    int width = 6, height = 6;
    Imgproc.morphologyEx(otsu, otsu, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(width+2,height+2)));
    saveFile(otsu, "findNumbers_otwarcie.jpg");
    Imgproc.morphologyEx(otsu, otsu, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(width+2,height+2)));
    saveFile(otsu, "findNumbers_zamkniecie.jpg");

    return null;
}
static void stdOtsu(Mat input, Mat output)
{
    Imgproc.threshold(input, output, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
}

static void morphOpen(Mat input, Mat output)
    {
        Imgproc.morphologyEx(input, output, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5)));
    }
    static void morphClose(Mat input, Mat output)
    {
        Imgproc.morphologyEx(input, output, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5)));
    }
    static void morphBH(Mat input, Mat output)
    {
        Imgproc.morphologyEx(input, output, Imgproc.MORPH_BLACKHAT, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(31,31)));
    }
    static void morphTH(Mat input, Mat output)
    {
        Imgproc.morphologyEx(input, output, Imgproc.MORPH_TOPHAT, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5)));
    }

static void findContours(Mat input, List<MatOfPoint> list)
{
    Imgproc.findContours(input, list, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
}
static void imagePlaying(Mat input)
{
    Mat gray = new Mat();
    toGray(input, gray);
    Mat otsu = new Mat();
    MatOfPoint bullseye = findBullsEye(input, otsu);
    MatOfPoint2f bullPoly = new MatOfPoint2f();
    Imgproc.approxPolyDP(new MatOfPoint2f(bullseye.toArray()), bullPoly, 1, true);

    Rect bullRect = Imgproc.boundingRect(bullseye);
    Point center = new Point();
    center.x = (bullRect.br().x + bullRect.tl().x) * 0.5;
    center.y = (bullRect.br().y + bullRect.tl().y) * 0.5;

    bullseye.release();
    List<Point> bullList = null;
    bullList = analyzePoints(bullPoly.toList(),center);
    bullPoly.release();
    System.out.println(bullList.size());

    double distanceW = bullRect.width/10;
    double distanceH = bullRect.height/10;
    bullseye.fromList(bullList);
    System.out.println("Nowa elipsa: " + bullseye);
    RotatedRect rect = simpleEllipse(bullseye);
    Imgproc.ellipse(input, rect, new Scalar(0,0,255), 3);
    for(int i=0; i<6; ++i)
    {
        rect.size.width += 2.7*distanceW;
        rect.size.height += 2.7*distanceH;
        Imgproc.ellipse(input, rect, new Scalar(0,0,255), 3);
    }
    saveFile(input, "znowElipsy.jpg");
}

static void BH(Mat input)
{
    Mat gray = new Mat();
    Mat BH = new Mat();
    toGray(input, gray);
    morphBH(gray, BH);
    saveFile(BH, "black_hat.jpg");
    Mat otsu = new Mat();
    stdOtsu(BH, otsu);
    saveFile(otsu, "black_hat_otsu.jpg");
    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    Imgproc.findContours(otsu,contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
    System.out.println("BH - liczba konturow: " + contours.size());

    Point[] points = new Point[4];
    List<Double> indicators = new ArrayList<Double>();
    double S, R1;
    for(int i=0; i<contours.size(); ++i)
    {
        MatOfPoint2f converted = new MatOfPoint2f(contours.get(i).toArray());
        RotatedRect rect = Imgproc.minAreaRect(converted);
        S = Math.abs(Imgproc.contourArea(converted, true));
        R1 = 2 * S / Math.PI;
        indicators.add(R1);
        System.out.println(i+": "+indicators.get(i));
    }
    //Collections.max(indicators);

    //int index = biggest(indicators);
    //System.out.println("To jest ten indeks: " + index);
    //Imgproc.drawContours(BH, contours, index, new Scalar(0,0,255), -1);
    //Collections.sort(indicators, Collections.reverseOrder());
    ArrayList<Integer> indexes = new ArrayList<Integer>();
    Imgproc.cvtColor(otsu,otsu,Imgproc.COLOR_GRAY2BGR);
    otsu.convertTo(otsu,CvType.CV_8UC3);
    for(int i=0;i<10;++i) {
        int index = biggest(indicators);

        indexes.add(index);
        System.out.println(index);
        Imgproc.drawContours(otsu, contours, index, new Scalar(0,0,255), 1);
        RotatedRect box = Imgproc.fitEllipse(new MatOfPoint2f(contours.get(index).toArray()));
        Imgproc.ellipse(input, box, new Scalar(0,0,255),3);
        System.out.println("Wartosc: " + indicators.get(index));
        indicators.set(index, -1.0);
    }

    saveFile(input,"BH_elipsy.jpg");
    saveFile(otsu, "kontur.jpg");
}

static void fromCenter(Point center, Mat input)
{
    Mat gray = new Mat();
    toGray(input, gray);
    Mat derivative = ScharrDer(gray);
    Mat otsu = new Mat();
    stdOtsu(derivative, derivative);
    Imgproc.drawMarker(derivative, center, Scalar.all(255));
    Mat mask = new Mat(input.rows(), input.cols(), CvType.CV_8UC1);


    Imgproc.line(mask, center, new Point(0,0), Scalar.all(255));


    saveFile(mask,"rynek.jpg");
    saveFile(derivative, "rynek0.jpg");
}
static void myCanny(Mat input, Point center)
{
    Mat gray = new Mat();
    toGray(input, gray);
    Imgproc.medianBlur(gray, gray, 5);
    Core.MinMaxLocResult result = Core.minMaxLoc(gray);
    System.out.println("Result: " + result.maxVal);
    double max = 50;
    double min = max/2;
    Mat canny = new Mat();
    Imgproc.Canny(gray, canny, min, max);
    saveFile(canny, "canny.jpg");

    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    Imgproc.findContours(canny, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
    for(MatOfPoint e: contours)
    {
        if(e.toList().size() > 20) {

            RotatedRect rect = simpleEllipse(e);
            //Imgproc.drawMarker(input, rect.center, new Scalar(0, 0, 255));
            if(euclidDst(center, rect.center) < 50) {
                Imgproc.ellipse(input, rect, new Scalar(0, 0, 255), 1);
            }

        }
    }
    saveFile(input, "Srodki.jpg");
}
static void derTest(Mat input)
{
    Mat gray = new Mat();
    toGray(input, gray);
    Mat rKx = new Mat(2,2, CvType.CV_32SC1);
    Mat rKy = new Mat(rKx.rows(), rKx.cols(), CvType.CV_32SC1);

    int[] value = {1};
    rKx.put(0,0, value);
    value[0] = 0;
    rKx.put(0,1, value);
    rKx.put(1,0, value);
    value[0] = -1;
    rKx.put(1,1, value);
    System.out.println("Roberts: " + rKx.dump());

    value[0] = 0;
    rKy.put(0,0, value);
    value[0] = 1;
    rKy.put(0,1, value);
    value[0] = -1;
    rKy.put(1,0, value);
    value[0] = 0;
    rKy.put(1,1, value);
    System.out.println("Roberts: " + rKy.dump()) ;

    Mat robertsX = new Mat(), robertsY = new Mat(), roberts = new Mat();
    Imgproc.filter2D(gray, robertsX, CvType.CV_8UC1, rKx);
    Imgproc.filter2D(gray, robertsY, CvType.CV_8UC1, rKy);
    //Core.addWeighted(robertsX, 0.5, robertsY, 0.5,0.0, roberts);
    //Core.bitwise_not(roberts,roberts);
    Core.add(robertsX, robertsY, roberts);
    saveFile(roberts, "KrzyzR.jpg");
    Mat otsu = new Mat();
    stdOtsu(roberts, otsu);
    saveFile(otsu, "KrzyzR_otsu.jpg");

    Mat P1 = new Mat(3,3,CvType.CV_32SC1);
    Mat P2 = new Mat(3,3,CvType.CV_32SC1);
    Mat P3 = new Mat(3,3,CvType.CV_32SC1);
    Mat P4 = new Mat(3,3,CvType.CV_32SC1);
    P1.setTo(Scalar.all(0));
    P2.setTo(Scalar.all(0));
    P3.setTo(Scalar.all(0));
    P4.setTo(Scalar.all(0));

    value[0] = -1;
    P1.put(0,0,value);
    P1.put(1,0,value);
    P1.put(2,0,value);

    P2.put(1,0,value);
    P2.put(2,0,value);
    P2.put(2,1,value);

    P3.put(2,0,value);
    P3.put(2,1,value);
    P3.put(2,2,value);

    P4.put(2,1,value);
    P4.put(2,2,value);
    P4.put(1,2,value);

    value[0] = 1;
    P1.put(0,2,value);
    P1.put(1,2,value);
    P1.put(2,2,value);

    P2.put(0,1,value);
    P2.put(0,2,value);
    P2.put(1,2,value);

    P3.put(0,0,value);
    P3.put(0,1,value);
    P3.put(0,2,value);

    P4.put(0,0,value);
    P4.put(1,0,value);
    P4.put(0,1,value);
    System.out.println("P1: " + P1.dump() + "\nP2: " + P2.dump() + "\nP3: " + P3.dump() + "\nP4: " + P4.dump());
    Mat Prewitt = new Mat(), PrewA = new Mat(), PrewB = new Mat(), PrewC = new Mat(), PrewD = new Mat();

    Imgproc.filter2D(gray, PrewA, CvType.CV_8UC1, P1);
    Imgproc.filter2D(gray, PrewB, CvType.CV_8UC1, P2);
    Imgproc.filter2D(gray, PrewC, CvType.CV_8UC1, P3);
    Imgproc.filter2D(gray, PrewD, CvType.CV_8UC1, P4);

    Core.add(PrewA, PrewB, Prewitt);
    Core.add(PrewC, Prewitt, Prewitt);
    Core.add(PrewD, Prewitt, Prewitt);
    saveFile(Prewitt, "Prewitt.jpg");
    stdOtsu(Prewitt, otsu);
    saveFile(otsu, "Prewitt_otsu.jpg");

    Mat sobel = new Mat();
    Imgproc.Sobel(gray, sobel, CvType.CV_8UC1, 1, 1);
    saveFile(sobel, "Sobel.jpg");
    stdOtsu(sobel, otsu);
    saveFile(otsu, "Sobel_otsu.jpg");

    Mat scharr = ScharrDer(gray);
    saveFile(scharr, "Scharr.jpg");
    stdOtsu(scharr, otsu);
    analyzeScharr(otsu);
    saveFile(otsu, "Scharr_otsu.jpg");

    //myCanny(input);
}
static void analyzeScharr(Mat otsu)
{
    Mat closed = new Mat();
    Imgproc.morphologyEx(otsu, closed, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(3,3)));
    saveFile(closed, "Scharr_otsu_closed.jpg");
}
static void analyzeOtsu(Mat otsu, Point center, Mat input)
{
    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    Imgproc.findContours(otsu, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
    Imgproc.drawMarker(input, center, new Scalar(0,0,255));
    List<Point> current = null;
    RotatedRect currentRect = null;
    List<Point> promising;
    for(int i=0; i<contours.size(); ++i)
    {
        current = contours.get(i).toList();
        if(current.size() >= 50)
        {
            currentRect = simpleEllipse(contours.get(i));
            if(euclidDst(center, currentRect.center) <=90.0) {
                promising = analyzePoints(current, currentRect.center);
                if(promising.size() >=5) {
                    MatOfPoint converted = new MatOfPoint();
                    converted.fromList(promising);
                    currentRect = simpleEllipse(converted);
                    Imgproc.ellipse(input, currentRect, new Scalar(0, 0, 255), 3);
                    Imgproc.drawMarker(input, currentRect.center, new Scalar(255, 0, 0));
                }
            }
        }else
        {

        }
    }
    saveFile(input, "ell.jpg");
}

static void transforms()
{
    Mat input = openFile("WZORCE1.jpg");
    saveFile(input, "wzorzec.jpg");
    MatOfPoint pattern = findBullsEye(input, new Mat());
    input = openFile("S30.jpg");

    double factorx, factory;
    if (input.cols() < input.rows()) {
        factorx = 512.0 / input.cols();
        factory = 512.0 / input.rows();
    } else {
        factorx = 512.0 / input.cols();
        factory = 512.0 / input.rows();
    }

    Imgproc.resize(input, input, new Size(), factorx, factory, Imgproc.INTER_LANCZOS4); // Tutaj nastepuje redukcja rozdzielczosci
    MatOfPoint image = findBullsEye(input, new Mat());
    System.out.println("Pattern: " + pattern);
    System.out.println("Image: " + image);
    saveFile(input, "przeskalowany.jpg");
    Mat dupa = new Mat();
    List<Point> array = new ArrayList<Point>();
    List<Point> array2 = new ArrayList<Point>();
    List<Point> imageList = image.toList();
    List<Point> patternList = pattern.toList();
    int range = 3;
    array.add(imageList.get(0));array.add(imageList.get(1));array.add(imageList.get(2));
    array2.add(patternList.get(0));array2.add(patternList.get(1));array2.add(patternList.get(2));
    MatOfPoint2f fromImage = new MatOfPoint2f();
    fromImage.fromList(array);
    MatOfPoint2f fromPattern = new MatOfPoint2f();
    fromPattern.fromList(array2);
    Mat M = Imgproc.getAffineTransform(fromImage, fromPattern);
    Imgproc.warpAffine(input, dupa, M,new Size(512,512));
    saveFile(dupa,"afiniczne.jpg");
}
static double Xv;
static double Yv;
static String additional;
    public static void main(String[] args) {

        System.out.println("Poczatek programu, wersja OpenCV: " + Core.VERSION);
        /*double liczba = 7, srednia=0.0;
        List<Double> X = new ArrayList<Double>();
        List<Double> Y = new ArrayList<Double>();
        X.add(4128.0); X.add(3264.0); X.add(4128.0);X.add(3264.0);X.add(1024.0);
        Y.add(3096.0); Y.add(2448.0); Y.add(2322.0);Y.add(1836.0);Y.add(768.0);
        for(int a=1; a<=X.size(); ++a) {
            for (int i = 1; i <= liczba; ++i) {
                Mat input = openFile("CM25_" + i + ".jpg");
                additional = "CM25";
                Xv = X.get(a-1);
                Yv = Y.get(a-1);
                counter = i;
                double factorx = 0;
                double factory = 0;

                if (input.cols() < input.rows()) {
                    factorx = Y.get(a-1) / input.cols();
                    factory = X.get(a-1) / input.rows();
                } else {
                    factorx = 1024.0 / input.cols();
                    factory = 768.0 / input.rows();
                }

                Imgproc.resize(input, input, new Size(), factorx, factory, Imgproc.INTER_LANCZOS4); // Tutaj nastepuje redukcja rozdzielczosci/*

                double[] size = {0.0};
                Compose.findBullsEye(input, new Mat(), size);
                System.out.println("Rozmiar "+X.get(a-1)+"x"+Y.get(a-1)+": " + size[0]);
                srednia += size[0];
            }
            srednia /= liczba;
            System.out.println("Srednia: " + srednia);
        }*/


        ++counter;
        int licznik = 0;
        int dolny = 1;
        int gorny = 81;
        List<Integer> bledy = new ArrayList<Integer>();

        for(int i=dolny;i<=gorny;++i) {
            counter = i;
            Mat input = openFile("S"+i+".jpg");
            double factorx = 0;
            double factory = 0;
            if(input.cols()/input.rows() != 1) {
                if (input.cols() < input.rows()) {
                    factorx = 768.0 / input.cols();
                    factory = 1024.0 / input.rows();
                } else {
                    factorx = 1024.0 / input.cols();
                    factory = 768.0 / input.rows();
                }

                Imgproc.resize(input, input, new Size(), factorx, factory, Imgproc.INTER_LANCZOS4); // Tutaj nastepuje redukcja rozdzielczosci
            }else
            {
                input = Resolution.changeRes(input, 1000, 1000);
            }
            System.out.println(input.rows() + "\t" + input.cols());
            saveFile(input, "aktualny.jpg");

            Strategy strategy = new StrategyNumber();
            Compose compose = new Compose(strategy, input);

            try {
                if(compose.launch() <= 0)
                {
                    bledy.add(i);
                }
            }catch(RuntimeException e)
            {
                System.out.println("!!!!!!!!!!!!!!!!!!Blad czasu wykonania dla:" + i);
                e.printStackTrace();
                ++licznik;
            }

        }
        System.out.println("Tyle bledow naliczono: "+ licznik);
        for(int i: bledy)
        {
            System.out.print(i +"\t");
        }
    }
}
