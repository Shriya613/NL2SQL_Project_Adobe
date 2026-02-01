BEGIN TRANSACTION;
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE,
    Nationality TEXT,
    Biography TEXT);
INSERT INTO "Authors" VALUES(1,'AuthorA','1970-01-01','American','Biography of Author A');
INSERT INTO "Authors" VALUES(2,'AuthorB','1972-02-02','English','Biography of Author B');
INSERT INTO "Authors" VALUES(3,'AuthorC','1974-03-03','Canadian','Biography of Author C');
INSERT INTO "Authors" VALUES(4,'AuthorD','1976-04-04','Australian','Biography of Author D');
INSERT INTO "Authors" VALUES(5,'AuthorE','1978-05-05','Irish','Biography of Author E');
CREATE TABLE Novels(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Author TEXT,
    Year INTEGER,
    Genre TEXT,
    Character TEXT);
INSERT INTO "Novels" VALUES(1,'Book1','AuthorA',2000,'Fiction','John Smith');
INSERT INTO "Novels" VALUES(2,'Book2','AuthorB',2002,'Mystery','Jane Doe');
INSERT INTO "Novels" VALUES(3,'Book3','AuthorC',2004,'Romance','Robert Johnson');
INSERT INTO "Novels" VALUES(4,'Book4','AuthorD',2006,'Thriller','Lisa Brown');
INSERT INTO "Novels" VALUES(5,'Book5','AuthorE',2008,'Science Fiction','David Miller');
CREATE TABLE Publishers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    EstablishedYear INTEGER,
    Location TEXT,
    NumberOfEmployees INTEGER);
INSERT INTO "Publishers" VALUES(1,'PublisherX',1980,'New York',1000);
INSERT INTO "Publishers" VALUES(2,'PublisherY',1990,'London',2000);
INSERT INTO "Publishers" VALUES(3,'PublisherZ',2000,'Toronto',3000);
INSERT INTO "Publishers" VALUES(4,'PublisherW',2010,'Dublin',4000);
INSERT INTO "Publishers" VALUES(5,'PublisherV',2020,'Tokyo',5000);
CREATE TABLE ShortStories(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Author TEXT,
    Year INTEGER,
    Theme TEXT,
    Character TEXT);
INSERT INTO "ShortStories" VALUES(1,'Story1','AuthorA',2001,'Adventure','John Smith');
INSERT INTO "ShortStories" VALUES(2,'Story2','AuthorB',2003,'Humor','James White');
INSERT INTO "ShortStories" VALUES(3,'Story3','AuthorC',2005,'Drama','Elizabeth Green');
INSERT INTO "ShortStories" VALUES(4,'Story4','AuthorD',2007,'Tragedy','Tommy Black');
INSERT INTO "ShortStories" VALUES(5,'Story5','AuthorE',2009,'Horror','William Gray');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Novels',5);
INSERT INTO "sqlite_sequence" VALUES('ShortStories',5);
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Publishers',5);
COMMIT;
