BEGIN TRANSACTION;
CREATE TABLE Articles(
    ArticleID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    PublicationDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    WordCount INTEGER,
    MagazineId INTEGER REFERENCES Magazines(MagazineID),
    AuthorId INTEGER REFERENCES Authors(AuthorID));
INSERT INTO "Articles" VALUES(1,'Article One','2024-02-19 15:09:29',1000,1,1);
INSERT INTO "Articles" VALUES(2,'Article Two','2024-02-19 15:09:29',1500,2,2);
INSERT INTO "Articles" VALUES(3,'Article Three','2024-02-19 15:09:29',2000,3,3);
INSERT INTO "Articles" VALUES(4,'Article Four','2024-02-19 15:09:29',2500,4,4);
INSERT INTO "Articles" VALUES(5,'Article Five','2024-02-19 15:09:29',3000,5,5);
CREATE TABLE Authors(
    AuthorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    EmailAddress TEXT UNIQUE);
INSERT INTO "Authors" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Authors" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Authors" VALUES(3,'Michael','Brown','michael.brown@example.com');
INSERT INTO "Authors" VALUES(4,'Emily','White','emily.white@example.com');
INSERT INTO "Authors" VALUES(5,'David','Black','david.black@example.com');
CREATE TABLE "Documentaries"(
    DocumentaryId INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseYear INTEGER ,
    Creator TEXT,
    LengthInMinutes INTEGER,
    MagazineId INTEGER REFERENCES Magazines(MagazineID));
INSERT INTO "Documentaries" VALUES(1,'Doc One',2001,'Producer One',60,1);
INSERT INTO "Documentaries" VALUES(2,'Doc Two',2002,'Producer Two',90,2);
INSERT INTO "Documentaries" VALUES(3,'Doc Three',2003,'Producer Three',120,3);
INSERT INTO "Documentaries" VALUES(4,'Doc Four',2004,'Producer Four',150,4);
INSERT INTO "Documentaries" VALUES(5,'Doc Five',2005,'Producer Five',180,5);
CREATE TABLE Magazines(
    MagazineID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    YearEstablished INTEGER ,
    PublisherId INTEGER REFERENCES Publishers(PublisherID));
INSERT INTO "Magazines" VALUES(1,'Time',1923,1);
INSERT INTO "Magazines" VALUES(2,'National Geographic',1888,2);
INSERT INTO "Magazines" VALUES(3,'People',1974,3);
INSERT INTO "Magazines" VALUES(4,'Men''s Health',1987,4);
INSERT INTO "Magazines" VALUES(5,'USA Today',1982,5);
CREATE TABLE Publishers(
    PublisherID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT);
INSERT INTO "Publishers" VALUES(1,'Time Inc.','New York, USA');
INSERT INTO "Publishers" VALUES(2,'Conde Nast','London, UK');
INSERT INTO "Publishers" VALUES(3,'Hearst Corporation','New York, USA');
INSERT INTO "Publishers" VALUES(4,'Rodale Inc.','Emmaus, Pennsylvania, USA');
INSERT INTO "Publishers" VALUES(5,'Gannett Company','McLean, Virginia, USA');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Publishers',5);
INSERT INTO "sqlite_sequence" VALUES('Magazines',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Documentaries',5);
COMMIT;
