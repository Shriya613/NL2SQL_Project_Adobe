BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Date_of_Birth DATE,
    Ethnicity TEXT);
INSERT INTO "Actor" VALUES(1,'Leonardo DiCaprio','11/11/1974','German-Irish-Italian');
INSERT INTO "Actor" VALUES(2,'Tom Hardy','15/09/1977','English');
INSERT INTO "Actor" VALUES(3,'Brad Pitt','18/12/1963','American');
INSERT INTO "Actor" VALUES(4,'Cate Blanchett','14/05/1969','Australian');
INSERT INTO "Actor" VALUES(5,'Scarlett Johansson','22/11/1984','American');
CREATE TABLE "Award" (
    id INTEGER PRIMARY KEY,
    Awards TEXT,
    Film TEXT,
    Year INTEGER,
    Presenter TEXT);
INSERT INTO "Award" VALUES(1,'Best Picture','The Irishman',2019,'Robert De Niro');
INSERT INTO "Award" VALUES(2,'Best Supporting Actress','Once Upon a Time...in Hollywood',2019,'Margot Robbie');
INSERT INTO "Award" VALUES(3,'Best Animated Feature','Toy Story 4',2019,'Tim Allen');
INSERT INTO "Award" VALUES(4,'Best Cinematography','Joker',2019,'Zack Snyder');
INSERT INTO "Award" VALUES(5,'Best Original Score','It Chapter Two',2019,'Bill Skarsgård');
CREATE TABLE Director (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Birthdate DATE,
    Nationality TEXT);
INSERT INTO "Director" VALUES(1,'Christopher Nolan','20/07/1970','American-British');
INSERT INTO "Director" VALUES(2,'Quentin Tarantino','27/03/1963','American');
INSERT INTO "Director" VALUES(3,'Martin Scorsese','17/11/1942','American');
INSERT INTO "Director" VALUES(4,'Guillermo del Toro','09/12/1964','Mexican');
INSERT INTO "Director" VALUES(5,'Denis Villeneuve','03/04/1967','Canadian');
CREATE TABLE Movie (
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Release_Date DATE,
    Runtime INTEGER,
    Genre TEXT);
INSERT INTO "Movie" VALUES(1,'Inception','16/07/2010',148,'Science Fiction');
INSERT INTO "Movie" VALUES(2,'Interstellar','07/11/2014',169,'Science Fiction');
INSERT INTO "Movie" VALUES(3,'Dunkirk','21/07/2017',106,'War');
INSERT INTO "Movie" VALUES(4,'Tenet','31/08/2020',150,'Action');
INSERT INTO "Movie" VALUES(5,'Man of Steel','14/06/2013',143,'Superhero');
CREATE TABLE Producer (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Production_Company TEXT,
    Number_Of_Productions INTEGER);
INSERT INTO "Producer" VALUES(1,'David Heyman','Heyday Films',50);
INSERT INTO "Producer" VALUES(2,'Barbara Broccoli','EON Productions',26);
INSERT INTO "Producer" VALUES(3,'Albert Brooks','Brooksfilms',20);
INSERT INTO "Producer" VALUES(4,'Kevin Feige','Marvel Studios',50);
INSERT INTO "Producer" VALUES(5,'Frank Marshall','Amblin Entertainment',50);
CREATE TABLE Viewership (
    id INTEGER PRIMARY KEY,
    Movie_ID INTEGER REFERENCES Movie(id),
    Total_Views INTEGER,
    Peak_Concurrent_Streams INTEGER,
    Platform TEXT);
INSERT INTO "Viewership" VALUES(1,1,1000000,500000,'Netflix');
INSERT INTO "Viewership" VALUES(2,2,750000,300000,'Hulu');
INSERT INTO "Viewership" VALUES(3,3,500000,200000,'Amazon Prime Video');
INSERT INTO "Viewership" VALUES(4,4,250000,100000,'Disney+');
INSERT INTO "Viewership" VALUES(5,5,300000,150000,'Apple TV+');
COMMIT;
