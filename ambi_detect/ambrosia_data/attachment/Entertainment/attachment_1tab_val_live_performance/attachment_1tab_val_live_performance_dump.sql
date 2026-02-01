BEGIN TRANSACTION;
CREATE TABLE Artist(
    ArtistID INTEGER PRIMARY KEY,
    Name TEXT,
    Genre TEXT
);
INSERT INTO "Artist" VALUES(1,'Band A','Pop');
INSERT INTO "Artist" VALUES(2,'Band B','Rock');
INSERT INTO "Artist" VALUES(3,'Band C','Jazz');
INSERT INTO "Artist" VALUES(4,'Band D','Country');
INSERT INTO "Artist" VALUES(5,'Band E','Hip Hop');
CREATE TABLE Artists_in_Concert(
    ConcertID INTEGER,
    ArtistID INTEGER,
    FOREIGN KEY (ConcertID) REFERENCES Concert(ConcertID),
    FOREIGN KEY (ArtistID) REFERENCES Artist(ArtistID),
    PRIMARY KEY (ConcertID, ArtistID)
);
INSERT INTO "Artists_in_Concert" VALUES(1,1);
INSERT INTO "Artists_in_Concert" VALUES(2,2);
INSERT INTO "Artists_in_Concert" VALUES(3,3);
INSERT INTO "Artists_in_Concert" VALUES(4,4);
INSERT INTO "Artists_in_Concert" VALUES(5,5);
CREATE TABLE Concert(
    ConcertID INTEGER PRIMARY KEY,
    Band_or_Artist_name TEXT,
    Album TEXT
);
INSERT INTO "Concert" VALUES(1,'Band X','Album Y');
INSERT INTO "Concert" VALUES(2,'Band Z','Album W');
INSERT INTO "Concert" VALUES(3,'Band AA','Album BB');
INSERT INTO "Concert" VALUES(4,'Band AB','Album BA');
INSERT INTO "Concert" VALUES(5,'Band AC','Album CA');
CREATE TABLE Live_Event(
    EventID INTEGER PRIMARY KEY,
    Title TEXT,
    Date DATE,
    Time TIME,
    VenueName TEXT,
    Live_Performance TEXT ,
    Venue_Capacity INT
);
INSERT INTO "Live_Event" VALUES(1,'Rock Concert','2021-12-31','19:00:00','Citi Field','Concert',500);
INSERT INTO "Live_Event" VALUES(2,'Classical Music Night','2021-12-31','18:00:00','Carnegie Hall','Concert',600);
INSERT INTO "Live_Event" VALUES(3,'A Midsummer Night''s Dream','2022-05-27','19:00:00','Central Park','Theater Play',500);
INSERT INTO "Live_Event" VALUES(4,'Hamlet','2022-05-27','18:00:00','Broadway Theatre','Theater Play',700);
CREATE TABLE Plays_at_Venues(
    PlayID INTEGER,
    VenueName TEXT,
    FOREIGN KEY (PlayID) REFERENCES Theater_Play(PlayID),
    FOREIGN KEY (VenueName) REFERENCES Live_Event(VenueName),
    PRIMARY KEY (PlayID, VenueName)
);
INSERT INTO "Plays_at_Venues" VALUES(1,'Central Park');
INSERT INTO "Plays_at_Venues" VALUES(2,'Broadway Theatre');
INSERT INTO "Plays_at_Venues" VALUES(3,'Central Park');
INSERT INTO "Plays_at_Venues" VALUES(4,'Broadway Theatre');
INSERT INTO "Plays_at_Venues" VALUES(5,'Central Park');
CREATE TABLE Theater_Play(
    PlayID INTEGER PRIMARY KEY,
    Title TEXT,
    Author TEXT,
    Year INT
);
INSERT INTO "Theater_Play" VALUES(1,'Othello','Shakespeare',1603);
INSERT INTO "Theater_Play" VALUES(2,'Macbeth','Shakespeare',1606);
INSERT INTO "Theater_Play" VALUES(3,'King Lear','Shakespeare',1606);
INSERT INTO "Theater_Play" VALUES(4,'Antony and Cleopatra','Shakespeare',1607);
INSERT INTO "Theater_Play" VALUES(5,'Twelfth Night','Shakespeare',1602);
COMMIT;
