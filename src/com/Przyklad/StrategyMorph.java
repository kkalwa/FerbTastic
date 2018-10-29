package com.Przyklad;


import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;


public class StrategyMorph extends Strategy {

    /**
     * Funkcja sprawdza 'rozpietosc' danego konturu. Przyjmuje kontur, punkt srodkowy tarczy oraz liczbe cwiartek
     * przez ktore kontur musi przechodzic. Funkcja zaklada ze w punkcie srodkowym znajduje sie srodek ukladu wspolrzednych
     * nastepnie dla kazdego punktu z konturu sprawdzane jest w ktorej cwiartce tego ukladu wspolrzednych sie znajduje.
     * Punkty sa pozniej przypisywane do odpowiedniej listy (kazda cwiartka ma swoja liste) i gdy dany kontur przechodzi
     * przez liczbe cwiartek okreslona przez condition to zwracana jest wartosc true
     *
     * @param contour   - kontur, ktory poddajemy analizie
     * @param center    - punkt, ktory uznajemy za srodek tarczy
     * @param condition - liczba okreslajaca ile cwiartek ukladu wsporzednych kontur powinien przecinac
     * @return true - kontur pomyslnie przeszedl test rozpietosci; false - kontur nie przeszedl pomyslnie testu rozpietosci
     */
    static boolean isOk(MatOfPoint contour, Point center, int condition) {
        List<Point> conList = contour.toList(); // Utworzenie listy punktow. Nazwa 'conList' jest troche nieczytelna :(
        boolean output = false;   // Ten obiekt nalezy zwrocic na koncu funkcji. True oznacza, ze dany kontur przeszedl klasyfikacje, false przeciwnie
        List<Point> TL = new ArrayList<Point>(); // Top Left - punkty w lewym gornym rogu
        List<Point> TR = new ArrayList<Point>(); // Top Right - punkty w prawym gornym rogu
        List<Point> BL = new ArrayList<Point>(); // Bottom Left - punkty w lewym dolnym rogu
        List<Point> BR = new ArrayList<Point>(); // Bottom Right - punkty w prawym dolnym rogu
        int count = 3; // Okresla ile punktow dla kazdego rogu nalezy zebrac
        int full = 0; // Gdy ktoras z list sie zapelni to nalezy powiekszyc ten obiekt o jeden

        for (Point p : conList)  // Iteracja po wszystkich punktach konturu
        {
            if (p.x < center.x) {
                if ((p.y < center.y) && (TL.size() < count)) // Lezy w lewym gornym rogu
                {
                    if (TL.size() == count - 1) {
                        ++full;
                    }
                    TL.add(p);
                } else if ((p.y > center.y) && (BL.size() < count)) // Lezy w lewym dolnym rogu
                {
                    if (BL.size() == count - 1) {
                        ++full;
                    }
                    BL.add(p);
                }

            } else if (p.x > center.x) {
                if ((p.y < center.y) && (TR.size() < count)) // Lezy w prawym gornym rogu
                {
                    if (TR.size() == count - 1) {
                        ++full;
                    }
                    TR.add(p);
                } else if ((p.y > center.y) && (BR.size() < count)) // Lezy w prawym dolnym rogu
                {
                    if (BR.size() == count - 1) {
                        ++full;
                    }
                    BR.add(p);
                }
            }
        }

        if (full >= condition) // Sprawdzenie ile cwiartek sie zapelnilo
        {
            output = true;
        }
        return output;
    }

    /**
     * Funkcja przyjmuje liste wykrytych elips, w ktorej znajduje sie wiele elips zdublowanych i usuwa te zdublowane.
     * Sposob klasyfikacji odbywa sie poprzez porownanie pola dwoch sasiednich elips
     *
     * @param ellipses - lista wykrytych elips
     * @return Lista bez zdublowanych elips
     */
    static List<RotatedRect> deleteDoubles(List<RotatedRect> ellipses) {
        RotatedRect current = null; // aktualna elipsa
        RotatedRect next = null; // poprzednia elipsa
        List<RotatedRect> output = new ArrayList<RotatedRect>(); // Lista elips ktora zwracamy
        double difference = 10000.0;   // roznica miedzy rozmiarami sasiednich elips


        for (int i = 0; i < ellipses.size() - 1; ++i) {
            current = ellipses.get(i);
            next = ellipses.get(i + 1);

            if ((Math.abs(next.size.width - current.size.width) > 30) && (Math.abs(next.size.height - current.size.height) > 30)) {
                output.add(current);
            }

        }
        output.add(ellipses.get(ellipses.size() - 1));

        return output;
    }

public static List<MatOfPoint> linkContours(List<MatOfPoint> contours, Point center)
{
    List<Double> distance = new ArrayList<Double>();
    for(MatOfPoint c: contours)
    {
        List<Double> distances = Compose.euclidDist(c.toList(), center);
        double max = Collections.max(distances);
        distance.add(max);
    }
    double difference = 25.0;
    List<List<Integer>> integers = new ArrayList<List<Integer>>();
    for(int i=0; i<distance.size(); ++i)
    {
        List<Integer> current = new ArrayList<Integer>();
        for(int j=i; j<distance.size(); ++j)
        {

            if(distance.get(j) != -difference && (distance.get(j) > distance.get(i) - difference) && (distance.get(j) < distance.get(i) + difference))
            {
                current.add(j);
            }
        }
        if(current.size() > 0) {
            integers.add(current);
            for(int e: current)
            {
                distance.remove(e);
                distance.add(e, -difference);
            }
        }
    }

    List<MatOfPoint> newCont = new ArrayList<MatOfPoint>();
    for(List<Integer> i: integers)
    {
        List<Point> combined = new ArrayList<Point>();
        for(int index: i)
        {
            combined.addAll(contours.get(index).toList());

        }
        MatOfPoint aux = new MatOfPoint();
        aux.fromList(combined);
        newCont.add(aux);
    }
return newCont;
}
    public static List<Point> indicators(Compose launcher, Mat input, Point center, MatOfPoint contour) {
        List<Double> distances = launcher.euclidDist(contour.toList(), center);
        List<Point> points = contour.toList();
        List<Point> output = new ArrayList<Point>();
        MatOfDouble dists = new MatOfDouble();
        dists.fromList(distances);
        double[] mean = Core.mean(dists).val;
        System.out.println("Srednia z Core: " + mean[0]);
        double[] min = {0.0};
        double[] max = {0.0};
        launcher.minMax(distances, min, max);
        //System.out.println("Min: " + min[0] + "\tMax: " + max[0]);
        double alfa = 0.1;
        int i = 0;
        for (Double p : distances) {
            if (((p > 0.75 * mean[0]) && (p < 1.25 * mean[0]))) {
                output.add(points.get(i));
            }
            ++i;
        }
        return output;
    }

    public static void indicators2(List<Point> list, Point center, Mat input) {
        List<Double> dists = new ArrayList<Double>();
        for (Point p : list) {
            dists.add(Compose.euclidDst(center, p));
        }
        double mean = simpleAvg(dists);
        double max = simpleMax(dists);
        double min = simpleMin(dists);
        double current = 0.0, kappa = 0.0;
        for (int i = 0; i < dists.size(); ++i) {
            current = dists.get(i);
            kappa = (current - mean) / (max - min);
            System.out.println(kappa);
            if (kappa < 0.4) {
                Imgproc.circle(input, list.get(i), 10, new Scalar(255, 0, 0), 3);
            } else if (kappa > 0.6) {
                Imgproc.circle(input, list.get(i), 10, new Scalar(0, 0, 255), 3);
            } else {
                Imgproc.circle(input, list.get(i), 10, new Scalar(0, 255, 0), 3);
            }
        }
    }

    /**
     * Funkcja oblicza srednia z listy liczb typu Double.
     *
     * @param list - lista liczb typu Double
     * @return srednia z liczb
     */
    public static double simpleAvg(List<Double> list) {
        double avg = 0.0;
        int size = list.size();
        for (double e : list) {
            avg += e;
        }
        avg = avg / size;
        return avg;
    }

    /**
     * Funkcja znajduje wartosc maksymalna z listy liczb typu Double
     *
     * @param list - lista liczb typu Double
     * @return wartosc maksymalna
     */
    public static double simpleMax(List<Double> list) {
        double max = list.get(0);
        for (double e : list) {
            if (e > max) {
                max = e;
            }
        }
        return max;
    }

    /**
     * Funkcja znajduje wartosc minimalna z listy liczb typu Double
     *
     * @param list - lista liczb typu Double
     * @return Najmniejsza wartosc z listy
     */
    public static double simpleMin(List<Double> list) {
        double min = list.get(0);
        for (double e : list) {
            if (e < min) {
                min = e;
            }
        }
        return min;
    }

    /**
     * Glowna funkcja obliczajaca punktacje. Jest to funkcja wirtualna, dziedziczona z klasy {@link Strategy}.
     * Wykonuje kolejno czynnosci:
     * 1. Wstepnie lokalizuje centralne kolo tarczy na podstawie wskaznika cyrkularnosci
     * 2. Wstepnie lokalizuje srodek tarczy jako srodek konturu zwiazanego z centralnym kolem
     * 3. Na podstawie  polozenia srodka odrzucone zostaja punkty nie lezace na obwodzie kola
     * 4. Po odrzuceniu blednych punktow ponownie zostaje obliczony punkt centralny
     * 5. Detekcja elips tarczy odbywa sie z wykorzystaniem operacji morfologicznych Top Hat i Black Hat i funkcji findContours() z OpenCV
     * 6. Bledy typu false positive sa odrzucane wykorzystujac odleglosc od srodka tarczy, rozmiar konturu i kryterium rozpietosci
     * 7. Przestrzeliny sa wykryte przez progowanie wykrywajace najwieksze wartosci jasnosci na obrazie
     * 8. Do przestrzelin dopasowywane sa okregi
     * 9 Wyliczana jest punktacja na podstawie powyzszych danych
     * @param launcher - obiekt typu Compose, ktory wywoluje funkcje
     * @return liczba zdobytych punktow
     */
    /**
     * Funkcja lokalizuje centralne biale kolo
     *
     * @param bullseye - kontur centralnego kola
     * @param gray     - obraz z ktorego wycinamy centralne kolo
     * @return obiekt typu RotatedRect reprezentujacy elipse w centralnej czesci tarczy
     */
    public RotatedRect whiteCircle(MatOfPoint bullseye, Mat gray, Point center) {
        Mat mask = new Mat(gray.rows(), gray.cols(), CvType.CV_8UC1);
        mask.setTo(Scalar.all(0));
        List<MatOfPoint> list = new ArrayList<MatOfPoint>();
        list.add(bullseye); // bullseye nalezy umiescic w liscie
        Imgproc.drawContours(mask, list, -1, Scalar.all(255), -1); // dopiero bullseye zapisany w liscie moze byc narysowany
        FerbTastic.saveFile(mask, "maska.jpg");
        Mat cropped = new Mat();
        gray.copyTo(cropped, mask);  // Tutaj nastepuje wyciecie centralnego kola i zapisanie go w cropped
        FerbTastic.saveFile(cropped, "bullseye.jpg");
        Mat otsu = new Mat();
        //Compose.stdOtsu(cropped, otsu);
        Imgproc.threshold(cropped, otsu, 120, 255, Imgproc.THRESH_BINARY);
        FerbTastic.saveFile(otsu, "otsuBialy.jpg");
        Imgproc.morphologyEx(otsu, otsu, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(11, 11)));
        Imgproc.morphologyEx(otsu, otsu, Imgproc.MORPH_DILATE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5)));
        FerbTastic.saveFile(otsu, "otsuBialy.jpg");
        list.clear();
        Imgproc.findContours(otsu, list, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
        MatOfPoint2f current = new MatOfPoint2f();
        for (MatOfPoint e : list) {
            Imgproc.approxPolyDP(new MatOfPoint2f(e.toArray()), current, 1, true);
            if (Imgproc.pointPolygonTest(current, center, false) == 1) {
                break;
            }
        }

        List<Point> points = Compose.analyzePoints(current.toList(), center, false);
        //FerbTastic.drawPoints(gray, points);
        FerbTastic.saveFile(gray, "test.jpg");
        MatOfPoint2f output = new MatOfPoint2f();
        output.fromList(points);
        return Imgproc.fitEllipse(output);
    }

    /**
     * Funkcja dokona sortowania uzyskanych elips od najmniejszej do najwiekszej
     *
     * @param target - lista wykrytych elips
     * @return
     */
    public static void sortTarget(List<RotatedRect> target) {
        List<Double> areas = new ArrayList<Double>(); // Klasyfikacja odbedzie sie na podstawie pol poszczegolnych elips

        for (RotatedRect e : target) // obliczenie pol
        {
            areas.add(e.size.area());
        }

        for (int i = 0; i < areas.size() - 1; ++i) // sortowanie wlasciwe (zgodne z algorytmem babelkowym)
        {
            for (int j = areas.size() - 1; j > i; --j) {
                if (areas.get(j) < areas.get(j - 1)) {
                    areas.add(j - 1, areas.get(j));
                    areas.remove(j + 1);
                    target.add(j - 1, target.get(j));
                    target.remove(j + 1);
                }
            }
        }

    }

    @Override
    public double computeScore(Compose launcher) {
        Mat gray = launcher.grayscale;
        Mat input = launcher.image;
        Mat output = new Mat();

        double x = input.cols() / 147.0, y = input.rows() / 196.0; // dane do zmniejszenia rozdzielczosci
        if( (int)(x%2) == 0) // jesli x jest parzyste to nalezy zmniejszyc o 1
            x = x-1;
        if( ((int)y % 2) == 0) // jesli y parzysty to zmniejszamy o 1
            y = y-1;
        System.out.println(x + "\n" + y);
        Imgproc.GaussianBlur(gray, gray, new Size(x, y), 5); // Na poczatku nastepuje mocne rozmycie obrazu
        //FerbTastic.saveFile(gray, FerbTastic.counter+"rozmyty.jpg");
        //FerbTastic.saveFile(gray, "blurred.jpg");
        List<RotatedRect> target = new ArrayList<RotatedRect>(); // Lista elips, ktore tworza tarcze
        double[] size = {0.0}; // Rozmiar bullseye, potrzebny do okreslania odleglosci na tarczy. Jest tablica aby mozna bylo go przeslac jako argument - wynik
        MatOfPoint bullseye = launcher.findBullsEye(input, output, size); // przeszuka obraz w poszukiwaniu bullseye i zwroci kontur go reprezentujacy oraz jako argument-wynik binarke oraz rozmiar
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

        double dist = size[0] / 49.0; // Rozmiar bullseye to 49 mm. Dziele euklidesowa odleglosc przez ten rozmiar
        //approxDistances(input, bullseye, dist, center);
        //areas(input, dist, center);

        RotatedRect white = whiteCircle(bullseye, gray, center); // Poszukiwanie centralnego bialego kola
        Mat mask = new Mat(gray.rows(), gray.cols(), CvType.CV_8UC1);
        mask.setTo(Scalar.all(0));
        Imgproc.ellipse(mask, white, Scalar.all(255));

        //FerbTastic.saveFile(gray, FerbTastic.counter+"czarny_srodek.jpg");
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(mask, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
        MatOfPoint2f polyWhite = new MatOfPoint2f();
        Imgproc.approxPolyDP(new MatOfPoint2f(contours.get(1).toArray()), polyWhite, 1, true);
        contours.clear();
        mask.setTo(Scalar.all(0));

        center = white.center; // Okregi najblizsze srodkowi najlepiej oddaja polozenie punktu centralnego, wiec po raz kolejny center jest zmieniany
        Imgproc.drawMarker(input, center, new Scalar(0, 255, 0));
        FerbTastic.saveFile(input, "zielony_srodek.jpg");
        target.add(white); // white powinien byc pierwszy na tej liscie
        target.add(launcher.simpleEllipse(bullseye));

        int counter = 0;
        Mat otsu = new Mat();
        Mat BH = new Mat();
        Imgproc.morphologyEx(gray, BH, Imgproc.MORPH_BLACKHAT, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7, 7))); // Na poczatku bylo 11

        FerbTastic.saveFile(BH, "BH.jpg");
        Imgproc.threshold(BH, otsu, 3, 255, Imgproc.THRESH_BINARY);
        FerbTastic.saveFile(otsu, FerbTastic.counter + "BH_otsu.jpg");

        contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(otsu, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);


        System.out.println("Znaleziono: " + contours.size() + " konturow");
        int radius = (int) (dist * 8.0);
        for (MatOfPoint e : contours) {
            promising = Compose.analyzePoints(e.toList(), center, true);
            if (promising.size() > 20) {

                RotatedRect rect = launcher.simpleEllipse(e);
                Imgproc.drawMarker(input, rect.center, new Scalar(0, 0, 255));

                if ((launcher.euclidDst(center, rect.center) < radius) && (isOk(e, center, 3) == true)) {
                    target.add(rect);
                    ++counter;
                }

            }
        }
        sortTarget(target);
        mask.setTo(Scalar.all(0));
        Imgproc.ellipse(mask, target.get(target.size()-1), Scalar.all(255), -1);

        Mat interesting = new Mat(); // Tutaj bedzie czesc tarczy, ktora nas interesuje
        interesting.setTo(Scalar.all(0));
        gray.copyTo(interesting, mask); // W tym miejsu odbywa sie kopiowanie czesci obrazu
        FerbTastic.saveFile(interesting, "wyciety.jpg");
        mask.release();

        Mat histogram = launcher.stdHist(interesting);
        System.out.println(histogram);
        float[] value = {0};
        histogram.put(1,0, value);
        Core.MinMaxLocResult result = Core.minMaxLoc(histogram);
        System.out.println("Histogram: " + result.maxLoc.y);
        Mat brightest = new Mat();
        brightest.setTo(Scalar.all(0));
        double offset = 10;
        Imgproc.threshold(interesting, brightest, /*result.maxLoc.y + offset*/ 215, 255, Imgproc.THRESH_BINARY);
        Imgproc.morphologyEx(brightest, brightest, Imgproc.MORPH_DILATE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3,3)));
        FerbTastic.saveFile(brightest, FerbTastic.counter + "bright.jpg");
        histogram.release();
        Imgproc.ellipse(interesting, white, Scalar.all(0), -1); // zamalowanie centralnego bialego kola

        Mat TH = new Mat();
        Imgproc.morphologyEx(interesting, TH, Imgproc.MORPH_TOPHAT, Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(11, 11))); // Uzyskanie Top Hat
        TH.setTo(Scalar.all(0), brightest);
        FerbTastic.saveFile(TH, FerbTastic.counter+"TH.jpg");


        //Compose.stdOtsu(TH, otsu);  // Top Hat uzyskany wczesniej zostaje zbinaryzowany metoda otsu
        //simpleBin(TH, otsu, launcher, 30);
        Imgproc.threshold(TH, otsu, 3, 255, Imgproc.THRESH_BINARY);

        //Imgproc.morphologyEx(otsu, otsu, Imgproc.MORPH_DILATE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3))); // Dylatacja, aby pogrubić krawedzie
        FerbTastic.saveFile(otsu, FerbTastic.counter+"TH_otsu.jpg");
        contours.clear();
        Imgproc.findContours(otsu, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

        /*List<MatOfPoint> linked = linkContours(contours, center);
        int a = 30;
        int b = a+1;
        Mat copy = new Mat();

        for(int i=0; i<linked.size(); ++i) {
            input.copyTo(copy);
            MatOfPoint e = linked.get(i);
            FerbTastic.drawPoints(copy, e.toList());
            FerbTastic.saveFile(copy, i+"early.jpg");
        }*/


        System.out.println("Znaleziono: " + contours.size() + " konturow");
         radius = (int) (dist * 8.0); // Okreslenie promienia dla okregu w ktorym musza znalezc sie srodki elips
        Imgproc.drawMarker(input, center, new Scalar(255, 0, 0)); // na niebiesko zaznaczono punkt srodkowy, ktory wynika z bullseye (nie jest tozsamy z srodkiem tarczy!)


        for (MatOfPoint e : contours) {
            promising = Compose.analyzePoints(e.toList(), center, true);
            //promising = indicators(launcher, input, center, e);
            if (promising.size() > 10) {

                RotatedRect rect = launcher.simpleEllipse(e);

                if ((launcher.euclidDst(center, rect.center) < radius) && (isOk(e, center, 3) == true)) {
                    target.add(rect);
                    ++counter;
                }

            }
        }

        contours.clear();
        /* W tym miejscu byl black hat wczesniej*/
        sortTarget(target);
        if (target.size() > 0) {
            target = deleteDoubles(target);
        } else {
            System.out.println("Nie znaleziono tarczy!");
            throw new RuntimeException();
        }
        sortTarget(target);
        //int licznik = 1;
        //Mat aux = new Mat();
        for (RotatedRect e : target) {
            //launcher.image.copyTo(aux);
            Imgproc.ellipse(launcher.image, e, new Scalar(0, 0, 255), 3);
            //++licznik;
        }

        FerbTastic.saveFile(launcher.image, FerbTastic.counter+"Srodki.jpg"); // To potem przeniesc
        System.out.println("Counter: " + counter);


        FerbTastic.saveFile(gray, "klo.jpg");


        // Nalezy usunac calkowicie centralny okrag. Nie wiadomo gdzie w target sie on znajduje, dlatego najpierw sortuje target
        sortTarget(target);
        // majac posortowane elipsy mozna usunac najmniejsza z nich, jednak nigdy nie wiadomo czy zostala ona wykryta i nalezy to sprawdzic
        int ellipseCount = 11; // liczba elips w danym typie tarczy (uwzgledniajac te najmniejsza)
        boolean everything; // czy wszystkie zostaly wykryte
        if (target.size() == ellipseCount) {
            everything = true; // Jesli wykryto 11 (jedenascie) elips to znaczy ze udalo sie zidentyfikowac cala tarcze
            target.remove(0); // Skoro wykryto wszystkie elipsy, to najmniejsza z nich jest ta ktora chcemy usunac
        } else // jesli nie wykryto calej tarczy to nalezy zastosowac inne podejscie
        {
            double contains = 0;
            Point[] points = new Point[4];
            target.get(0).points(points);
            contains = Imgproc.pointPolygonTest(polyWhite, points[0], false); // sprawdzam czy najmniejsza elipsa jest wewnatrz bialego kola
            if (contains == 1) // jesli tak to odrzucam
            {
                target.remove(0);
            } // jesli nie to po prostu nic sie nie robi
        }
        mask = new Mat(input.rows(), input.cols(), CvType.CV_8UC1);
        mask.setTo(Scalar.all(0));
        Compose.drawEllipses(target, mask);
        FerbTastic.saveFile(mask, "target2.jpg"); // w target2.jpg umieszcza sie elipsy pierscieni
        contours.clear(); // oczyszczenie listy contours, ktora i tak zaraz zostanie zapelniona innymi konturami

        Imgproc.findContours(mask, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE); // odczyt z mask konturow zwiazanych z pierscieniami i zapisanie ich do contours
        mask.release();


        // Przechodzimy do obrobki przestrzelin
        // w contours sa kontury pierscieni
        List<MatOfPoint> lightAreas = new ArrayList<MatOfPoint>();
        Imgproc.findContours(brightest, lightAreas, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE); // lightAreas to lista wykrytych przestrzelin (bez tla)
        List<Integer> weights = new ArrayList<Integer>();
        List<Point> centers = StrategyRes.analyzeHoles(lightAreas, weights); // W tym momencie obliczone sa srodki znalezionych przestrzelin. Trzeba jeszcze okreslic ich srednice
        double r = (4.5 * dist) / 2.0; //kaliber to 4.5 mm, wiec przemnazam; zmienna oznacza promien okregow przyblizajacych przestrzeliny
        mask = new Mat(gray.rows(), gray.cols(), CvType.CV_8UC1);
        mask.setTo(Scalar.all(0));
        for (Point p : centers)  // Tutaj zapisuje sie okregi przyblizajace przestrzeliny
        {
            Imgproc.circle(mask, p, (int) (r), Scalar.all(255), -1);
        }
        lightAreas.clear();
        Imgproc.findContours(mask, lightAreas, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE); // od teraz w lightAreas sa okregi przybizajace przestrzeliny
        FerbTastic.saveFile(mask, FerbTastic.counter + "holes.jpg");
        mask.release();
        brightest.release();

        List<MatOfPoint2f> targetPoly = new ArrayList<MatOfPoint2f>(); // w targetPoly beda przechowywane wielomiany pierscieni
        MatOfPoint2f currentPoly = new MatOfPoint2f();
        for (MatOfPoint p : contours) // W tej petli nastepuje konwersja zwyczajnych konturow (klasa MatOfPoint) do wielomianow (klasa MatOfPoint2f)
        {
            Imgproc.approxPolyDP(new MatOfPoint2f(p.toArray()), currentPoly, 1, true);
            targetPoly.add(new MatOfPoint2f(currentPoly));
        }
        contours.clear(); //ZAPAMIETAC!!!!!!!! Contours jest teraz puste zeby nie marnowac pamieci
        Point closest = null;
        List<Point> closestList = new ArrayList<Point>();
        for (MatOfPoint e : lightAreas) // W tej petli szuka sie punktu najblizszego do srodka tarczy
        {
            List<Point> eList = e.toList();
            double lowest = Compose.euclidDst(center, eList.get(0));
            closest = eList.get(0);
            for (Point p : eList) {
                double dst = Compose.euclidDst(center, p);
                if (dst < lowest) {
                    lowest = dst;
                    closest = p;
                }
            }

            closestList.add(new Point(closest.x, closest.y));


        }
        // w closestList znajduja sie punkty kazdego konturu ktore leza najblizej srodka
        double score = 0.0;
        int base = 10;

        for (int i = 1; i < targetPoly.size(); i += 2) // Podliczanie punktow
        {
            Iterator<Integer> weightIt = weights.iterator(); // Iterator dla wag
            MatOfPoint2f ring = targetPoly.get(i);
            if (closestList.isEmpty() == false) {
                for (Iterator<Point> it = closestList.iterator(); it.hasNext(); ) {
                    Point p = it.next(); // Pobranie punktu danego konturu
                    int w = weightIt.next(); // Pobranie wagi
                    if (Imgproc.pointPolygonTest(ring, p, false) == 1) //Jesli punkt lezy wewnatrz to dodajemy punktacje i usuwamy z listy
                    {
                        score += base*w;
                        it.remove(); // Bezpieczne usuniecie z listy po ktorej iterujemy moze sie odbyc tylko przez interfejs Iteratora
                        weightIt.remove();
                        System.out.println("Dodaje: " + base*w);
                    }
                }
            } else {
                break;
            }
            base -= 1;
        }
        System.out.println("Tyle punktow naliczono: " + score);
        Imgproc.putText(input, Double.toString(score), new Point(50,50), Core.FONT_HERSHEY_SIMPLEX,1.0, new Scalar(0,0,255), 2);
        FerbTastic.saveFile(input, FerbTastic.counter + "a.jpg");
        FerbTastic.saveFile(input, "punkty/"+FerbTastic.counter + "a.jpg");
        return score;
    }
}

