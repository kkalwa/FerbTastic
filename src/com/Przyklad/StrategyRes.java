package com.Przyklad;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class StrategyRes extends Strategy {

    static List<RotatedRect> segments2(Mat image, MatOfPoint bullseye, Point center)
    {
        Mat gray = new Mat();
        Compose.toGray(image, gray);    /*Szare zdjecie*/

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
        //saveFile(derivative, "SobelDerivative.jpg");

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
                L = Compose.euclidDst(center, current);           /*Odleglosc miedzy punktem srodkowym a punktem na obwodzie*/
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
                actualPoints = Compose.analyzePoints(ring, center, true);
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

        //saveFile(image, "depresja.jpg");
        return allRects;
    }

    public static List<Point> analyzeHoles(List<MatOfPoint> lightAreas, List<Integer> weights)
    {
        if(lightAreas.size() == 0)
        {
            throw new RuntimeException("Nie wykryto zadnych przestrzelin");
        }
        Mat forTesting = new Mat(1024, 768, CvType.CV_8UC1); // Na tym nalezy rysowac wyniki
        forTesting.setTo(Scalar.all(0));
        //Imgproc.drawContours(forTesting, lightAreas, -1, Scalar.all(255), -1);
        //FerbTastic.saveFile(forTesting,"p.jpg"); // Narysowanie przestrzelin tak tylko zeby miec pewnosc ze dane wejsciowe sa ok

        List<Point> centers = new ArrayList<Point>(); // Tutaj przechowywac srodki przestrzelin
        double avgArea = 0.0; // Srednia wartosc pola przestrzeliny
        for(MatOfPoint e: lightAreas)
        {
            avgArea += Imgproc.minAreaRect(new MatOfPoint2f(e.toArray())).size.area();
        }
        avgArea /= lightAreas.size();

        System.out.println("Srednia area: " + avgArea);

        for(MatOfPoint e: lightAreas)
        {
            double current = Imgproc.minAreaRect(new MatOfPoint2f(e.toArray())).size.area();
            double ratio = current/avgArea;
            if((int)ratio > 1) // Jesli stosunek jest wiekszy od jednosci to moze oznaczac ze przestrzelina jest zlozeniem kilku mniejszych przestrzelin
            {
                //Imgproc.drawMarker(forTesting, Imgproc.minAreaRect(new MatOfPoint2f(e.toArray())).center, new Scalar(0,255,0));
                //System.out.println("Znaleziono: " + (int)ratio + " tyle: " + ratio + " przekracza o tyle: " + (ratio - (int)ratio));
                int count = (int)ratio;
                if((ratio - count) > 0.5)
                {
                    ++count;
                }
                centers.add(Imgproc.minAreaRect(new MatOfPoint2f(e.toArray())).center);
                weights.add(count);
            }else
            {
                centers.add(Imgproc.minAreaRect(new MatOfPoint2f(e.toArray())).center);
                weights.add(1);
                //Imgproc.drawMarker(forTesting, Imgproc.minAreaRect(new MatOfPoint2f(e.toArray())).center, new Scalar(0,0,255));
            }
        }

        return centers;
    }

    public static double computeDistance(List<MatOfPoint> contours, Point center)
    {
        Mat forTesting = new Mat(1024,768,CvType.CV_8UC3); // Tutaj wypisywac wyniki POTEM USUNAC!!!!!!!!!!!!!!!!!!!!!!!!
        forTesting.setTo(Scalar.all(0));
        Imgproc.drawContours(forTesting, contours, 6, Scalar.all(255), 1);
        Imgproc.drawContours(forTesting, contours, 8, Scalar.all(255), 1);
        //FerbTastic.saveFile(forTesting, "computeDistance.jpg");
        double caliber = 45.0;        // kaliber podany w milimetrach

        System.out.println("Szosty ma tyle punktow: " + contours.get(6).toList().size());
        System.out.println("Osmy ma tyle punktow: " + contours.get(8).toList().size());

        double avgDist = 0.0;
        for(Point p: contours.get(6).toList())
        {
            avgDist += Compose.euclidDst(center, p);
        }
        avgDist /= contours.get(6).toList().size();

        double avgDist2 = 0.0;
        for(Point p: contours.get(8).toList())
        {
            avgDist2 += Compose.euclidDst(center, p);
        }
        avgDist2 /= contours.get(8).toList().size();


        avgDist = avgDist2 - avgDist;
        return avgDist/caliber;
    }

    static List<MatOfPoint> basedonMrp(Mat input, MatOfPoint bullseye, Point center)
    {
        Mat mask = new Mat(input.rows(), input.cols(), CvType.CV_8UC1); /*Sluzy do wyciecia centralnej czesci tarczy*/
        Mat gray = new Mat();
        Compose.toGray(input, gray);
        mask.setTo(Scalar.all(0));
        Imgproc.fillConvexPoly(mask, bullseye, Scalar.all(255));
        //saveFile(mask, "maska_bullseye.jpg");
        Mat cropped = new Mat();
        gray.copyTo(cropped, mask);
        //saveFile(cropped, "przyciety.jpg");
        Mat otsu = new Mat();
        Imgproc.threshold(cropped, otsu, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
        //saveFile(otsu, "cropped_otsu.jpg");
        Mat morphology = new Mat();
        Imgproc.morphologyEx(otsu,morphology, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(11,11)));
        //saveFile(morphology, "cropped_otwarty.jpg");
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
        //saveFile(morphology, "nieLubieSiebie.jpg"); // Otwarcie zeby powiekszyc nieco rozmiary przestrzelin i upodobnic je do okregow
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
        //saveFile(simple, "simple.jpg");
        contours.clear();
        Imgproc.findContours(simple, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        //System.out.println("W calym obrazie znaleziono tyle przestrzelin: " + contours.size());
        return contours;
    }


    @Override
    public double computeScore(Compose launcher) {
        Mat gray = launcher.grayscale;
        Mat input = launcher.image;
        Mat output = new Mat();

        Mat reduced = new Mat(); // Zredukowana rozdzielczosc
        double factorx = 768.0/input.cols();
        double factory = 1024.0/input.rows();
        Imgproc.resize(input, reduced, new Size(), factorx,factory, Imgproc.INTER_LANCZOS4); // Tutaj nastepuje redukcja rozdzielczosci
        System.out.println(reduced);
        //FerbTastic.saveFile(reduced, "reduced.jpg");

        // Ponizej obliczenia, konwersje i analiza prowadzace do uzyskania bullseye
        double[] size = {0.0};
        MatOfPoint bullseye = launcher.findBullsEye(reduced, output, size);
        System.out.println("Size po wyjsciu z funkcji: " + size[0]);
        //FerbTastic.saveFile(output, "slowLight.jpg");
        double dist = size[0]/49.0; // Rozmiar bullseye to 49 mm. Dziele euklidesowa odleglosc przez ten rozmiar

        MatOfPoint2f polyBulls = new MatOfPoint2f();
        MatOfPoint2f converted = new MatOfPoint2f(bullseye.toArray());
        Imgproc.approxPolyDP(converted, polyBulls, 1, true);
        Rect bullRect = Imgproc.boundingRect(bullseye);
        Point center = new Point();
        center.x = (bullRect.br().x + bullRect.tl().x) * 0.5;
        center.y = (bullRect.br().y + bullRect.tl().y) * 0.5; // Tutaj wyznaczenie srodka po raz pierwszy. Center bedzie wykorzystany do klasyfikacji punktow
        bullseye = new MatOfPoint(polyBulls.toArray());
        List<Point> bullList = bullseye.toList();
        List<Point> promising = Compose.analyzePoints(bullList, center, true);
        bullseye.fromList(promising);
        bullRect = Imgproc.boundingRect(bullseye);
        center.x = (bullRect.br().x + bullRect.tl().x) * 0.5;
        center.y = (bullRect.br().y + bullRect.tl().y) * 0.5; // Tutaj wyznaczenie srodka jeszcze raz, tym razem po

        List<RotatedRect> target = segments2(reduced, bullseye, center); // Tutaj nastepuje wykrycie pierscieni
        List<MatOfPoint> lightAreas = basedonMrp(reduced, bullseye, center); // Tutaj jasne punkty

        double avgX=0.0, avgY = 0.0; // Wartosci srednie punktu srodka tarczy
        Point avgCenter = new Point();
        for(int i=0; i<target.size(); ++i) // Obliczenia srednich srodka tarczy
        {
            RotatedRect current = target.get(i);
            avgX += current.center.x;
            avgY += current.center.y;
        }
        avgX /= target.size();
        avgY /= target.size();
        avgCenter.x = avgX;
        avgCenter.y = avgY; // Przypisanie srednich do punktu

        RotatedRect ellipse = null;
        Scalar kolor = new Scalar(0,0,255);
        for(int i=0;i<target.size(); ++i)
        {
            ellipse = target.get(i);
            Imgproc.ellipse(reduced,ellipse,kolor, 1); // Narysowanie elips na obraz output.jpg kolorem czerwonym
        }
        //FerbTastic.saveFile(reduced, "output.jpg"); // Zapisanie output.jpg
        Mat aux = new Mat();
        Core.inRange(reduced, kolor, kolor, aux);
        //FerbTastic.saveFile(aux,"wycieteElipsy.jpg"); //Wyciecie narysowanych wczesniej elips (teraz staja sie biale)

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        RotatedRect biggest = target.get(target.size()-1);
        Mat mask = new Mat(reduced.rows(), reduced.cols(), CvType.CV_8UC1);
        Imgproc.ellipse(mask, biggest, Scalar.all(255), -1); // Narysowanie na masce obszaru ktory bedzie brany pod uwage
        //FerbTastic.saveFile(mask, "najwieksze.jpg");
        Mat interesting = new Mat();
        Mat simple = new Mat(reduced.rows(), reduced.cols(), CvType.CV_8UC1);
        Imgproc.drawContours(simple, lightAreas, -1, Scalar.all(255),-1);
        //FerbTastic.saveFile(simple, "przestrzeliny.jpg"); // Na przestrzeliny.jpg sa wszystkie jasne obszary czyli nawet elementy tla
        simple.copyTo(interesting, mask); // Na tym etapie wycinany jest obszar o promieniu najwiekszego okregu !!!!!WAZNE!!!!!!!!!!

        Imgproc.morphologyEx(interesting, interesting, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(15,15)));
        Imgproc.morphologyEx(interesting, interesting, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(20,20)));
        //FerbTastic.saveFile(interesting, "przestrzeliny2.jpg"); // na tym obrazie sa jasne obszary znajdujace sie tylko na tarczy
        lightAreas.clear();
        Imgproc.findContours(interesting, lightAreas, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE); // lightAreas to lista wykrytych przestrzelin (bez tla)
        List<Integer> weights = new ArrayList<Integer>();
        List<Point> centers = analyzeHoles(lightAreas, weights); // W tym momencie obliczone sa srodki znalezionych przestrzelin. Trzeba jeszcze okreslic ich srednice

        contours.clear();
        Imgproc.findContours(aux, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE); // w contours sa kontury pierscieni
        System.out.println("Tyle przestrzelin po oczyszczeniu: " + lightAreas.size());
        System.out.println("LA: " + lightAreas.size()+"\nContours: "+contours.size());
        interesting.setTo(Scalar.all(0));

        double radius = (4.5*dist)/2.0; //kaliber to 4.5 mm, wiec przemnazam
        //Imgproc.drawContours(interesting, lightAreas, -1, Scalar.all(255),1);
        for(Point p: centers)
        {
            Imgproc.circle(interesting, p, (int)(radius), new Scalar(255,0,0), -1);
        }
        //FerbTastic.saveFile(interesting, "dopasowanie.jpg");
        lightAreas.clear();    // Usuniecie przestrzelin z lightContours
        Imgproc.findContours(interesting, lightAreas, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(aux,aux, Imgproc.COLOR_GRAY2BGR);
        List<MatOfPoint2f> targetPoly = new ArrayList<MatOfPoint2f>(); // w targetPoly beda przechowywane wielomiany pierscieni
        MatOfPoint2f currentPoly = new MatOfPoint2f();
        for(MatOfPoint p: contours) // W tej petli nastepuje konwersja zwyczajnych konturow (klasa MatOfPoint) do wielomianow (klasa MatOfPoint2f)
        {
            Imgproc.approxPolyDP(new MatOfPoint2f(p.toArray()), currentPoly, 1, true);
            targetPoly.add(new MatOfPoint2f(currentPoly));
        }
        contours.clear(); //ZAPAMIETAC!!!!!!!! Contours jest teraz puste zeby nie marnowac pamieci
        Point closest = null;
        List<Point> closestList = new ArrayList<Point>();
        for(MatOfPoint e: lightAreas) // W tej petli szuka sie punktu najblizszego do srodka tarczy
        {
            List<Point> eList = e.toList();
            double lowest = Compose.euclidDst(avgCenter, eList.get(0));
            for(Point p: eList)
            {
                double dst = Compose.euclidDst(avgCenter, p);
                if(dst<lowest)
                {
                    lowest = dst;
                    closest = p;
                }
            }

            closestList.add(new Point(closest.x,closest.y));
        }
        for(Point p: closestList) // Majac juz punkty najblizsze - stawia sie marker tam gdzie je wykryto
        {
            Imgproc.drawMarker(reduced, p, kolor);
        }
        //FerbTastic.saveFile(reduced,"najblizsze.jpg");

        double score = 0.0;
        int base = 10;
        for(int i=1; i<targetPoly.size(); i+=2) // Podliczanie punktow
        {
            MatOfPoint2f ring = targetPoly.get(i);
            if(closestList.isEmpty() == false)
            {
                for(Iterator<Point> it = closestList.iterator(); it.hasNext();)
                {
                    Point p = it.next();
                    if(Imgproc.pointPolygonTest(ring, p, false) == 1) //Jesli punkt lezy wewnatrz to dodajemy punktacje i usuwamy z listy
                    {
                        score += base;
                        it.remove();
                        System.out.println("Dodaje: " + base);
                    }
                }
            }else
            {
                break;
            }
            base -= 1;
        }
        System.out.println("Tyle punktow naliczono: " + score);

        return 0;
    }


}
