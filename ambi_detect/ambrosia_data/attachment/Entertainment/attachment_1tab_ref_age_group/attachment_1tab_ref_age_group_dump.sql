BEGIN TRANSACTION;
CREATE TABLE AgeGroup(
    ID INTEGER PRIMARY KEY,
    GroupName TEXT);
INSERT INTO "AgeGroup" VALUES(1,'Children''s Book');
INSERT INTO "AgeGroup" VALUES(2,'Young Adult Book');
INSERT INTO "AgeGroup" VALUES(3,'Teenage Book');
INSERT INTO "AgeGroup" VALUES(4,'Adult Book');
INSERT INTO "AgeGroup" VALUES(5,'Older Readers Book');
CREATE TABLE Author(
    ID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Biography TEXT);
INSERT INTO "Author" VALUES(1,'John','Doe','He is a famous author.');
INSERT INTO "Author" VALUES(2,'Jane','Smith','She is an illustrator.');
INSERT INTO "Author" VALUES(3,'Emily','Brown','She writes science fiction novels.');
INSERT INTO "Author" VALUES(4,'Michael','White','He has written many mystery books.');
INSERT INTO "Author" VALUES(5,'Linda','Green','She writes adventure stories.');
CREATE TABLE Books(
    ID INTEGER PRIMARY KEY,
    Title TEXT,
    Summary TEXT,
    ReleaseDate DATE,
    Price REAL,
    GenreID INTEGER,
    AgeGroupID INTEGER,
    AuthorID INTEGER,
    IllustratorID INTEGER,
    PublisherID INTEGER,
    FOREIGN KEY(GenreID) REFERENCES Genre(ID),
    FOREIGN KEY(AgeGroupID) REFERENCES AgeGroup(ID),
    FOREIGN KEY(AuthorID) REFERENCES Author(ID),
    FOREIGN KEY(IllustratorID) REFERENCES Illustrator(ID),
    FOREIGN KEY(PublisherID) REFERENCES Publisher(ID));
INSERT INTO "Books" VALUES(1,'Book A','Summary of book A.','2021-01-01',9.99,1,1,1,1,1);
INSERT INTO "Books" VALUES(2,'Book B','Summary of book B.','2021-02-01',12.99,2,1,2,1,2);
INSERT INTO "Books" VALUES(3,'Book C','Summary of book C.','2021-03-01',14.99,3,2,3,2,3);
INSERT INTO "Books" VALUES(4,'Book D','Summary of book D.','2021-04-01',10.99,4,1,4,1,4);
INSERT INTO "Books" VALUES(5,'Book E','Summary of book E.','2021-05-01',11.99,5,2,5,1,5);
CREATE TABLE Genre(
    ID INTEGER PRIMARY KEY,
    Name TEXT);
INSERT INTO "Genre" VALUES(1,'Fiction');
INSERT INTO "Genre" VALUES(2,'Nonfiction');
INSERT INTO "Genre" VALUES(3,'Mystery');
INSERT INTO "Genre" VALUES(4,'Science Fiction');
INSERT INTO "Genre" VALUES(5,'Adventure');
CREATE TABLE Illustrator(
    ID INTEGER PRIMARY KEY,
    Fullname TEXT);
INSERT INTO "Illustrator" VALUES(1,'Jane Smith');
INSERT INTO "Illustrator" VALUES(2,'Robert Johnson');
INSERT INTO "Illustrator" VALUES(3,'William Brown');
INSERT INTO "Illustrator" VALUES(4,'David White');
INSERT INTO "Illustrator" VALUES(5,'Mark Green');
CREATE TABLE Publisher(
    ID INTEGER PRIMARY KEY,
    CompanyName TEXT,
    Address TEXT,
    PhoneNumber TEXT);
INSERT INTO "Publisher" VALUES(1,'Random House','New York, USA','+1 212-598-6000');
INSERT INTO "Publisher" VALUES(2,'HarperCollins','London, UK','+44 207-867-8000');
INSERT INTO "Publisher" VALUES(3,'Macmillan Publishers','Berlin, Germany','+49 30-260-00-0');
INSERT INTO "Publisher" VALUES(4,'Simon & Schuster','New York, USA','+1 212-698-7000');
INSERT INTO "Publisher" VALUES(5,'Hachette Book Group','Paris, France','+33 1-55-62-20-00');
COMMIT;
