BEGIN TRANSACTION;
CREATE TABLE Authors(
    AuthorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    Country TEXT);
INSERT INTO "Authors" VALUES(1,'John','Doe','1980-06-30','USA');
INSERT INTO "Authors" VALUES(2,'Jane','Smith','1975-10-01','UK');
CREATE TABLE Editions(
    EditionID INTEGER PRIMARY KEY AUTOINCREMENT,
    ISBN TEXT,
    ReleaseDate DATE,
    Price DECIMAL(10,2),
    Quantity INT,
    LitID INTEGER,
    FOREIGN KEY(LitID) REFERENCES Literatures(LitID));
CREATE TABLE Literatures(
    LitID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Synopsis TEXT,
    Language TEXT,
    PublicationYear INT,
    LiteratureType TEXT ,
    AuthorID INTEGER,
    PublisherID INTEGER,
    FOREIGN KEY(AuthorID) REFERENCES Authors(AuthorID),
    FOREIGN KEY(PublisherID) REFERENCES Publishers(PublisherID));
INSERT INTO "Literatures" VALUES(1,'The Italian Novel','A story about an Italian adventure','Italian',2010,'Novel',1,1);
INSERT INTO "Literatures" VALUES(2,'The Spanish Novel','A tale of love and mystery','Spanish',2012,'Novel',2,2);
INSERT INTO "Literatures" VALUES(3,'La Poesia Italiana','An anthology of Italian poetry','Italian',2008,'Poem',1,1);
INSERT INTO "Literatures" VALUES(4,'El Pueblo Español','A collection of poems about Spain','Spanish',2011,'Poem',2,2);
CREATE TABLE Publishers(
    PublisherID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    PhoneNumber TEXT);
INSERT INTO "Publishers" VALUES(1,'ABC Publishing','123 Main St.','555-1234');
INSERT INTO "Publishers" VALUES(2,'XYZ Books','456 Elm St.','555-5678');
CREATE TABLE Reviews(
    ReviewID INTEGER PRIMARY KEY AUTOINCREMENT,
    Rating INT ,
    CommentText TEXT,
    ReviewerID INTEGER,
    EditionID INTEGER,
    FOREIGN KEY(ReviewerID) REFERENCES Authors(AuthorID),
    FOREIGN KEY(EditionID) REFERENCES Editions(EditionID));
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',2);
INSERT INTO "sqlite_sequence" VALUES('Publishers',2);
INSERT INTO "sqlite_sequence" VALUES('Literatures',4);
COMMIT;
