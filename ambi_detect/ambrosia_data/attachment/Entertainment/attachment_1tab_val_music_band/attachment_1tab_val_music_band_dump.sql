BEGIN TRANSACTION;
CREATE TABLE Albums(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Release_Date DATE,
    Label TEXT,
    Genre TEXT,
    FOREIGN KEY (Genre) REFERENCES Music_Bands(Name));
INSERT INTO "Albums" VALUES(1,'First Impressions','2000-06-15','Warner','Jazz');
INSERT INTO "Albums" VALUES(2,'Melodic Memories','2005-03-05','Universal','Jazz');
INSERT INTO "Albums" VALUES(3,'Strumming Strings','2002-02-20','EMI','Rock');
INSERT INTO "Albums" VALUES(4,'Heavy Metal Beats','2010-10-15','Columbia','Rock');
CREATE TABLE Concerts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Date DATE,
    Venue TEXT,
    Ticket_Price DECIMAL(7,2),
    Band_id INTEGER,
    FOREIGN KEY (Band_id) REFERENCES Music_Bands(id));
INSERT INTO "Concerts" VALUES(1,'2021-12-30','Madison Square Garden',100.5,1);
INSERT INTO "Concerts" VALUES(2,'2022-01-15','Wembley Arena',75.3,2);
INSERT INTO "Concerts" VALUES(3,'2021-11-20','Staples Center',85.75,3);
INSERT INTO "Concerts" VALUES(4,'2022-02-20','Sydney Opera House',120.9,4);
CREATE TABLE Fans(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    First_name TEXT,
    Last_name TEXT,
    Age INT,
    Email TEXT,
    Favorite_band_id INTEGER,
    FOREIGN KEY (Favorite_band_id) REFERENCES Music_Bands(id));
INSERT INTO "Fans" VALUES(1,'John','Smith',35,'johnsmith@email.com',1);
INSERT INTO "Fans" VALUES(2,'Anna','Doe',28,'annadoe@email.com',2);
INSERT INTO "Fans" VALUES(3,'Peter','Jackson',45,'pjackson@email.com',3);
INSERT INTO "Fans" VALUES(4,'Emma','White',32,'ewhite@email.com',4);
CREATE TABLE Music_Bands(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Genre TEXT,
    Founding_Year INT,
    Country TEXT);
INSERT INTO "Music_Bands" VALUES(1,'The Jazz Connection','Jazz',1985,'USA');
INSERT INTO "Music_Bands" VALUES(2,'Blue Note Ensemble','Jazz',1990,'Canada');
INSERT INTO "Music_Bands" VALUES(3,'Guitar Legends','Rock',1985,'UK');
INSERT INTO "Music_Bands" VALUES(4,'Vocal Vibes','Rock',1993,'Australia');
CREATE TABLE Reviews(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Rating INT ,
    Review_text TEXT,
    Song_id INTEGER,
    FOREIGN KEY (Song_id) REFERENCES Songs(id));
INSERT INTO "Reviews" VALUES(1,4,'Great song! Love the melody.',1);
INSERT INTO "Reviews" VALUES(2,5,'Amazing guitar work!',2);
INSERT INTO "Reviews" VALUES(3,3,'Good song but not my favorite.',3);
INSERT INTO "Reviews" VALUES(4,5,'Awesome lyrics and music!',4);
CREATE TABLE Songs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Length FLOAT,
    Album_id INTEGER,
    FOREIGN KEY (Album_id) REFERENCES Albums(id));
INSERT INTO "Songs" VALUES(1,'Night Flight',3.5,1);
INSERT INTO "Songs" VALUES(2,'Dreamy Skies',4.2,2);
INSERT INTO "Songs" VALUES(3,'Thunderous Riffs',3.8,3);
INSERT INTO "Songs" VALUES(4,'Acoustic Journey',4.1,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Music_Bands',4);
INSERT INTO "sqlite_sequence" VALUES('Albums',4);
INSERT INTO "sqlite_sequence" VALUES('Songs',4);
INSERT INTO "sqlite_sequence" VALUES('Concerts',4);
INSERT INTO "sqlite_sequence" VALUES('Fans',4);
INSERT INTO "sqlite_sequence" VALUES('Reviews',4);
COMMIT;
