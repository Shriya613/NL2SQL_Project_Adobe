BEGIN TRANSACTION;
CREATE TABLE Authors(
    AuthorId INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE,
    Country TEXT
);
INSERT INTO "Authors" VALUES(1,'John Doe','1980-01-01','USA');
INSERT INTO "Authors" VALUES(2,'Jane Smith','2000-06-15','UK');
INSERT INTO "Authors" VALUES(3,'Emily Johnson','1975-02-22','Canada');
INSERT INTO "Authors" VALUES(4,'Michael Brown','1990-10-28','Australia');
INSERT INTO "Authors" VALUES(5,'David Wilson','1985-11-14','New Zealand');
INSERT INTO "Authors" VALUES(6,'Alexander White','1992-04-03','Germany');
CREATE TABLE Books(
    BookId INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Literature TEXT,
    Rating REAL,
    AuthorId INTEGER,
    SubjectId INTEGER,
    FOREIGN KEY (AuthorId) REFERENCES Authors(AuthorId),
    FOREIGN KEY (SubjectId) REFERENCES Subjects(SubjectId)
);
INSERT INTO "Books" VALUES(1,'Math Mysteries','Fiction Book',3.5,1,1);
INSERT INTO "Books" VALUES(2,'A History of Mankind','Non-fiction Book',4.0,2,2);
INSERT INTO "Books" VALUES(3,'Life Forms','Fiction Book',2.5,3,3);
INSERT INTO "Books" VALUES(4,'Quantum Mechanics','Non-fiction Book',3.5,4,4);
INSERT INTO "Books" VALUES(5,'Ethical Questions','Fiction Book',4.0,5,5);
INSERT INTO "Books" VALUES(6,'Time Travel Theory','Non-fiction Book',2.5,6,6);
CREATE TABLE Reviews(
    ReviewId INTEGER PRIMARY KEY AUTOINCREMENT,
    Rating REAL,
    Comment TEXT,
    BookId INTEGER,
    FOREIGN KEY (BookId) REFERENCES Books(BookId)
);
INSERT INTO "Reviews" VALUES(1,3.5,'Great story!',1);
INSERT INTO "Reviews" VALUES(2,4.0,'Very informative.',2);
INSERT INTO "Reviews" VALUES(3,2.5,'Not what I expected...',3);
INSERT INTO "Reviews" VALUES(4,3.5,'Interesting read.',4);
INSERT INTO "Reviews" VALUES(5,4.0,'I learned a lot.',5);
INSERT INTO "Reviews" VALUES(6,2.5,'Could have been better.',6);
CREATE TABLE Subjects(
    SubjectId INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT
);
INSERT INTO "Subjects" VALUES(1,'Mathematics','The study of numbers, quantities, shapes and spaces.');
INSERT INTO "Subjects" VALUES(2,'History','The study of past events and their causes and effects.');
INSERT INTO "Subjects" VALUES(3,'Biology','The study of living organisms and life processes.');
INSERT INTO "Subjects" VALUES(4,'Physics','The study of matter and energy, and how they interact with one another.');
INSERT INTO "Subjects" VALUES(5,'Philosophy','The study of fundamental questions about existence, knowledge, values, reason, mind and language.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',6);
INSERT INTO "sqlite_sequence" VALUES('Subjects',5);
INSERT INTO "sqlite_sequence" VALUES('Books',6);
INSERT INTO "sqlite_sequence" VALUES('Reviews',6);
COMMIT;
