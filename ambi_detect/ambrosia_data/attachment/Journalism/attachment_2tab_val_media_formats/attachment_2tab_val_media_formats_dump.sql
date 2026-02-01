BEGIN TRANSACTION;
CREATE TABLE Articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    AuthorName TEXT,
    PublicationDate DATE);
INSERT INTO "Articles" VALUES(1,'News Article 1','John Doe','2023-01-01');
INSERT INTO "Articles" VALUES(2,'News Article 2','Jane Smith','2023-02-01');
INSERT INTO "Articles" VALUES(3,'News Article 3','Michael Johnson','2023-03-01');
INSERT INTO "Articles" VALUES(4,'News Article 4','Emily Brown','2023-04-01');
INSERT INTO "Articles" VALUES(5,'News Article 5','David Wilson','2023-05-01');
CREATE TABLE AudioDocuments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Description TEXT,
    Length INTEGER ,
    ReleaseDate DATE);
INSERT INTO "AudioDocuments" VALUES(1,'First Documentary','The history of journalism in detail.',70,'2023-01-01');
INSERT INTO "AudioDocuments" VALUES(2,'Second Documentary','A documentary about modern media.',80,'2023-02-01');
INSERT INTO "AudioDocuments" VALUES(3,'Third Documentary','Unraveling mysteries one episode at a time.',90,'2023-03-01');
INSERT INTO "AudioDocuments" VALUES(4,'Fourth Documentary','The future of journalism explored.',60,'2023-04-01');
INSERT INTO "AudioDocuments" VALUES(5,'Fifth Documentary','Investigating important issues worldwide.',50,'2023-05-01');
CREATE TABLE Authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE);
INSERT INTO "Authors" VALUES(1,'John','Doe','1980-01-01');
INSERT INTO "Authors" VALUES(2,'Jane','Smith','1985-02-01');
INSERT INTO "Authors" VALUES(3,'Michael','Johnson','1990-03-01');
INSERT INTO "Authors" VALUES(4,'Emily','Brown','1995-04-01');
INSERT INTO "Authors" VALUES(5,'David','Wilson','1998-05-01');
CREATE TABLE Newspapers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT);
INSERT INTO "Newspapers" VALUES(1,'Daily Times','New York City');
INSERT INTO "Newspapers" VALUES(2,'Metro News','London');
INSERT INTO "Newspapers" VALUES(3,'Herald Sun','Australia');
INSERT INTO "Newspapers" VALUES(4,'Los Angeles Gazette','California');
INSERT INTO "Newspapers" VALUES(5,'Canada Post','Toronto');
CREATE TABLE Podcasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Description TEXT,
    Length INTEGER ,
    ReleaseDate DATE);
INSERT INTO "Podcasts" VALUES(1,'First Podcast','This is the first podcast.',20,'2023-01-01');
INSERT INTO "Podcasts" VALUES(2,'Second Podcast','Another interesting podcast.',30,'2023-02-01');
INSERT INTO "Podcasts" VALUES(3,'Third Podcast','A long podcast on a variety of topics.',60,'2023-03-01');
INSERT INTO "Podcasts" VALUES(4,'Fourth Podcast','A short but insightful podcast.',15,'2023-04-01');
INSERT INTO "Podcasts" VALUES(5,'Fifth Podcast','Exploring new ideas and concepts.',45,'2023-05-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Podcasts',5);
INSERT INTO "sqlite_sequence" VALUES('AudioDocuments',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Newspapers',5);
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
COMMIT;
