BEGIN TRANSACTION;
CREATE TABLE Authors (
    AuthorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE,
    Country TEXT);
INSERT INTO "Authors" VALUES(1,'Jane Doe','1976-03-01','USA');
INSERT INTO "Authors" VALUES(2,'John Smith','1990-06-05','Canada');
INSERT INTO "Authors" VALUES(3,'Michael Johnson','1985-12-15','UK');
INSERT INTO "Authors" VALUES(4,'Emily Davis','1995-08-10','Australia');
INSERT INTO "Authors" VALUES(5,'David Brown','1980-01-05','New Zealand');
CREATE TABLE Books (
    BookID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    PageCount INTEGER,
    Book_Genre TEXT ,
    Year_Of_Publication INTEGER ,
    AuthorID INTEGER,
    PublisherID INTEGER,
    GenreID INTEGER,
    FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID),
    FOREIGN KEY (PublisherID) REFERENCES Publishers(PublisherID),
    FOREIGN KEY (GenreID) REFERENCES Genres(GenreID));
INSERT INTO "Books" VALUES(1,'The Science of Tomorrow',320,'Science Fiction Book',2018,1,2,1);
INSERT INTO "Books" VALUES(2,'Love Beyond Time',280,'Romance Book',2018,2,3,2);
INSERT INTO "Books" VALUES(3,'Into The Stars',350,'Science Fiction Book',2021,3,1,1);
INSERT INTO "Books" VALUES(4,'A Love Unforgettable',300,'Romance Book',2020,4,4,2);
INSERT INTO "Books" VALUES(5,'Mythical Creatures',250,'Fantasy Book',2017,5,5,3);
INSERT INTO "Books" VALUES(6,'The Haunting Shadows',260,'Horror Book',2019,6,2,4);
INSERT INTO "Books" VALUES(7,'The Art Of Deception',270,'Thriller Book',2020,7,4,5);
INSERT INTO "Books" VALUES(8,'History In Black And White',310,'Non-Fiction Book',2017,8,3,6);
INSERT INTO "Books" VALUES(9,'The Alien Encounter',290,'Science Fiction Book',2021,9,5,1);
INSERT INTO "Books" VALUES(10,'An Eternal Promise',285,'Romance Book',2018,10,1,2);
CREATE TABLE Genres (
    GenreID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT);
INSERT INTO "Genres" VALUES(1,'Science fiction books focus on imaginary elements such as futuristic science advanced technology.');
INSERT INTO "Genres" VALUES(2,'Fantasy genre consists of works based on myths and legends.');
INSERT INTO "Genres" VALUES(3,'Horror genre involves suspense and horror elements like ghosts and supernatural events.');
INSERT INTO "Genres" VALUES(4,'Thriller genre revolves around fast pacing plot filled with unexpected twists.');
INSERT INTO "Genres" VALUES(5,'Non-fiction genre deals with real life facts, events or people.');
CREATE TABLE Publishers (
    PublisherID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    YearEstablished INTEGER);
INSERT INTO "Publishers" VALUES(1,'Random House','NYC, USA',1927);
INSERT INTO "Publishers" VALUES(2,'HarperCollins','London, UK',1919);
INSERT INTO "Publishers" VALUES(3,'Simon & Schuster','NYC, USA',1924);
INSERT INTO "Publishers" VALUES(4,'Hachette Book Group','Paris, France',1942);
INSERT INTO "Publishers" VALUES(5,'Macmillan Publishers','London, UK',1843);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Publishers',5);
INSERT INTO "sqlite_sequence" VALUES('Genres',5);
INSERT INTO "sqlite_sequence" VALUES('Books',10);
COMMIT;
