BEGIN TRANSACTION;
CREATE TABLE Action_Movies(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Director TEXT,
    ReleaseYear INTEGER,
    Duration INT
);
INSERT INTO "Action_Movies" VALUES(1,'The Dark Knight','Christopher Nolan',2008,163);
INSERT INTO "Action_Movies" VALUES(2,'Inception','Christopher Nolan',2010,148);
INSERT INTO "Action_Movies" VALUES(3,'Mad Max: Fury Road','George Miller',2015,120);
INSERT INTO "Action_Movies" VALUES(4,'John Wick','Chad Stahelski',2014,131);
INSERT INTO "Action_Movies" VALUES(5,'Avatar','James Cameron',2009,178);
CREATE TABLE Actors(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    BirthDate TEXT,
    Nationality TEXT
);
INSERT INTO "Actors" VALUES(1,'Leonardo DiCaprio','11/11/1974','American');
INSERT INTO "Actors" VALUES(2,'Tom Hanks','07/09/1956','American');
INSERT INTO "Actors" VALUES(3,'Cate Blanchett','05/14/1969','Australian');
INSERT INTO "Actors" VALUES(4,'Kristen Stewart','04/09/1990','American');
INSERT INTO "Actors" VALUES(5,'Anne Hathaway','11/12/1982','American');
CREATE TABLE Awards(
    id INTEGER PRIMARY KEY,
    AwardType TEXT,
    RecipientId INTEGER,
    FOREIGN KEY(RecipientId) REFERENCES Actors(id) ON DELETE SET NULL
);
INSERT INTO "Awards" VALUES(1,'Best Actor',1);
INSERT INTO "Awards" VALUES(2,'Best Supporting Actor',2);
INSERT INTO "Awards" VALUES(3,'Best Actress',3);
INSERT INTO "Awards" VALUES(4,'Best Supporting Actress',4);
INSERT INTO "Awards" VALUES(5,'Best Picture',5);
CREATE TABLE Directors(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    BirthDate TEXT,
    Nationality TEXT
);
INSERT INTO "Directors" VALUES(1,'Christopher Nolan','07/30/1970','English');
INSERT INTO "Directors" VALUES(2,'George Miller','03/03/1942','Australian');
INSERT INTO "Directors" VALUES(3,'Chad Stahelski','06/04/1971','American');
INSERT INTO "Directors" VALUES(4,'Rob Reiner','03/29/1947','American');
INSERT INTO "Directors" VALUES(5,'Nora Ephron','06/19/1941','American');
CREATE TABLE Movie_Ratings(
    id INTEGER PRIMARY KEY,
    Rating TEXT,
    ReviewerName TEXT
);
INSERT INTO "Movie_Ratings" VALUES(1,'A+','Peter Travers');
INSERT INTO "Movie_Ratings" VALUES(2,'B','Owen Gleiberman');
INSERT INTO "Movie_Ratings" VALUES(3,'A-','Todd McCarthy');
INSERT INTO "Movie_Ratings" VALUES(4,'A','Michael Phillips');
INSERT INTO "Movie_Ratings" VALUES(5,'B+','Ann Hornaday');
CREATE TABLE Romantic_Comedies(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Director TEXT,
    ReleaseYear INTEGER,
    Duration INT
);
INSERT INTO "Romantic_Comedies" VALUES(1,'When Harry Met Sally','Rob Reiner',1989,100);
INSERT INTO "Romantic_Comedies" VALUES(2,'Sleepless In Seattle','Nora Ephron',1993,105);
INSERT INTO "Romantic_Comedies" VALUES(3,'Pretty Woman','Garry Marshall',1990,119);
INSERT INTO "Romantic_Comedies" VALUES(4,'Notting Hill','Roger Michell',1999,120);
INSERT INTO "Romantic_Comedies" VALUES(5,'Love Actually','Richard Curtis',2003,135);
COMMIT;
