BEGIN TRANSACTION;
CREATE TABLE Authors(
    AuthorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    Country TEXT);
INSERT INTO "Authors" VALUES(1,'John','Doe','1980-06-15','USA');
INSERT INTO "Authors" VALUES(2,'Jane','Smith','1978-10-30','UK');
CREATE TABLE Books(
    BookID INTEGER PRIMARY KEY,
    Title TEXT,
    PublicationYear INT,
    Pages INT,
    Educational_Materials TEXT,
    PublisherID INT,
    FOREIGN KEY (PublisherID) REFERENCES Publishers(PublisherID));
INSERT INTO "Books" VALUES(1,'Introduction to Mathematics',2021,300,'Textbook',1);
INSERT INTO "Books" VALUES(2,'Advanced Physics',2020,400,'Textbook',2);
INSERT INTO "Books" VALUES(3,'Digital Electronics',2021,350,'E-Book',1);
INSERT INTO "Books" VALUES(4,'Computer Networking Fundamentals',2020,450,'E-Book',2);
CREATE TABLE Publishers(
    PublisherID INTEGER PRIMARY KEY,
    CompanyName TEXT,
    Address TEXT,
    ContactNumber TEXT,
    Email TEXT);
INSERT INTO "Publishers" VALUES(1,'Pearson','New York','+1 (555) 123-4567','pearson@email.com');
INSERT INTO "Publishers" VALUES(2,'Wiley','Boston','+1 (555) 987-6543','wiley@email.com');
CREATE TABLE Subjects(
    SubjectID INTEGER PRIMARY KEY,
    Name VARCHAR(255));
INSERT INTO "Subjects" VALUES(1,'Mathematics');
INSERT INTO "Subjects" VALUES(2,'Physics');
COMMIT;
