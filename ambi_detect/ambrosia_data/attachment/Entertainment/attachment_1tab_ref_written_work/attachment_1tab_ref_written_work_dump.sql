BEGIN TRANSACTION;
CREATE TABLE Authors (
    AuthorID INTEGER PRIMARY KEY,
    Name TEXT,
    Country TEXT,
    BirthDate DATE
);
INSERT INTO "Authors" VALUES(1,'Dr. Emily Thompson','USA','1970-06-23');
INSERT INTO "Authors" VALUES(2,'John Doe','UK','1985-10-15');
INSERT INTO "Authors" VALUES(3,'Jane Smith','Canada','1990-07-27');
CREATE TABLE Publishers (
    PublisherID INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT
);
INSERT INTO "Publishers" VALUES(1,'Penguin Books','New York, USA');
INSERT INTO "Publishers" VALUES(2,'HarperCollins','London, UK');
INSERT INTO "Publishers" VALUES(3,'Doubleday','Toronto, Canada');
CREATE TABLE TypesOfWork (
    TypeID INTEGER PRIMARY KEY,
    Type TEXT UNIQUE
);
INSERT INTO "TypesOfWork" VALUES(1,'Essay');
INSERT INTO "TypesOfWork" VALUES(2,'Letter');
CREATE TABLE WorksPublishedByPublisher (
    WorkID INTEGER,
    PublisherID INTEGER,
    PublicationYear INTEGER,
    FOREIGN KEY (WorkID) REFERENCES WrittenWorks(WriteID),
    FOREIGN KEY (PublisherID) REFERENCES Publishers(PublisherID)
);
INSERT INTO "WorksPublishedByPublisher" VALUES(1,1,2021);
INSERT INTO "WorksPublishedByPublisher" VALUES(2,2,2021);
INSERT INTO "WorksPublishedByPublisher" VALUES(3,3,2021);
INSERT INTO "WorksPublishedByPublisher" VALUES(4,2,2021);
INSERT INTO "WorksPublishedByPublisher" VALUES(5,3,2021);
CREATE TABLE Writes (
    WriteID INTEGER PRIMARY KEY,
    AuthorID INTEGER,
    WorkID INTEGER,
    FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID),
    FOREIGN KEY (WorkID) REFERENCES WrittenWorks(WorkID)
);
INSERT INTO "Writes" VALUES(1,1,1);
INSERT INTO "Writes" VALUES(2,1,2);
INSERT INTO "Writes" VALUES(3,2,3);
INSERT INTO "Writes" VALUES(4,3,4);
INSERT INTO "Writes" VALUES(5,3,5);
CREATE TABLE WrittenWorks (
    WorkID INTEGER PRIMARY KEY,
    Title TEXT,
    TypeOfWork TEXT ,
    PublishedDate DATE,
    WordCount INTEGER,
    FOREIGN KEY (TypeOfWork) REFERENCES TypesOfWork(Type)
);
INSERT INTO "WrittenWorks" VALUES(1,'My Essay','Essay','2021-01-01',1000);
INSERT INTO "WrittenWorks" VALUES(2,'Important Letter','Letter','2021-05-01',500);
INSERT INTO "WrittenWorks" VALUES(3,'Another Essay','Essay','2021-02-01',800);
INSERT INTO "WrittenWorks" VALUES(4,'A Formal Letter','Letter','2021-04-01',600);
INSERT INTO "WrittenWorks" VALUES(5,'The Great Novel','Novel','2021-03-01',50000);
COMMIT;
