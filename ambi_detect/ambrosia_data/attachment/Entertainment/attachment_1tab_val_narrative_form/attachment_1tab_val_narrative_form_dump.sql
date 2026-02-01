BEGIN TRANSACTION;
CREATE TABLE Authors(
    AuthorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    Nationality TEXT
);
INSERT INTO "Authors" VALUES(1,'John','Doe','1960-01-01','American');
INSERT INTO "Authors" VALUES(2,'Jane','Smith','1987-03-22','British');
CREATE TABLE Books(
    BookID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Narrative_Form TEXT ,
    Epoch TEXT ,
    PageCount INT,
    PublicationYear YEAR,
    AuthorID INTEGER,
    GenreID INTEGER,
    FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID),
    FOREIGN KEY (GenreID) REFERENCES Genres(GenreID)
);
INSERT INTO "Books" VALUES(1,'My Life Story','Autobiography','20th Century',300,2000,1,1);
INSERT INTO "Books" VALUES(2,'The Other Side of the World','Autobiography','21st Century',280,2018,2,2);
INSERT INTO "Books" VALUES(3,'World War II History','Historical Account','20th Century',350,2010,3,3);
INSERT INTO "Books" VALUES(4,'Modern American Society','Historical Account','21st Century',400,2012,4,4);
CREATE TABLE Genres(
    GenreID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE
);
INSERT INTO "Genres" VALUES(1,'Biographies & Memoirs');
INSERT INTO "Genres" VALUES(2,'Fiction & Literature');
CREATE TABLE Published_Works(
    WorkID INTEGER PRIMARY KEY AUTOINCREMENT,
    BookID INTEGER,
    PublisherID INTEGER,
    ReleaseDate DATE,
    SalesAmount DECIMAL(19, 5),
    FOREIGN KEY (BookID) REFERENCES Books(BookID),
    FOREIGN KEY (PublisherID) REFERENCES Publishers(PublisherID)
);
INSERT INTO "Published_Works" VALUES(1,1,1,'2000-01-01',1000);
INSERT INTO "Published_Works" VALUES(2,2,2,'2018-06-01',1500);
INSERT INTO "Published_Works" VALUES(3,3,1,'2010-01-01',2000);
INSERT INTO "Published_Works" VALUES(4,4,2,'2012-06-01',2500);
CREATE TABLE Publishers(
    PublisherID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    YearEstablished INT
);
INSERT INTO "Publishers" VALUES(1,'AbeBooks','Abingdon Road, Abingdon, Oxfordshire OX14 3LY, UK',1998);
INSERT INTO "Publishers" VALUES(2,'Amazon Publishing','41800 West 10 Mile Rd., Ada, MI 49355',2009);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',2);
INSERT INTO "sqlite_sequence" VALUES('Genres',2);
INSERT INTO "sqlite_sequence" VALUES('Publishers',2);
INSERT INTO "sqlite_sequence" VALUES('Books',4);
INSERT INTO "sqlite_sequence" VALUES('Published_Works',4);
COMMIT;
