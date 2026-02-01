BEGIN TRANSACTION;
CREATE TABLE BooksInSubject (
    bookId INTEGER,
    subjectId INTEGER,
    FOREIGN KEY (bookId) REFERENCES Textbooks (id) ON DELETE SET NULL,
    FOREIGN KEY (subjectId) REFERENCES Subjects (id) ON DELETE SET NULL,
    PRIMARY KEY (bookId, subjectId)
);
INSERT INTO "BooksInSubject" VALUES(1,1);
INSERT INTO "BooksInSubject" VALUES(2,2);
INSERT INTO "BooksInSubject" VALUES(3,3);
INSERT INTO "BooksInSubject" VALUES(4,4);
INSERT INTO "BooksInSubject" VALUES(5,5);
CREATE TABLE EBooks (
    id INTEGER PRIMARY KEY,
    title TEXT,
    publisherId INTEGER,
    format TEXT ,
    price REAL ,
    FOREIGN KEY (publisherId) REFERENCES Publisher (id)
);
INSERT INTO "EBooks" VALUES(1,'Computer Science',4,'EPUB',27.99);
INSERT INTO "EBooks" VALUES(2,'Mathematics',4,'PDF',26.99);
INSERT INTO "EBooks" VALUES(3,'Engineering',5,'EPUB',25.99);
INSERT INTO "EBooks" VALUES(4,'Philosophy',5,'PDF',24.99);
INSERT INTO "EBooks" VALUES(5,'Art History',3,'MOBI',23.99);
CREATE TABLE Publisher (
    id INTEGER PRIMARY KEY,
    name TEXT
);
INSERT INTO "Publisher" VALUES(1,'Pearson');
INSERT INTO "Publisher" VALUES(2,'Wiley');
INSERT INTO "Publisher" VALUES(3,'McGraw-Hill');
INSERT INTO "Publisher" VALUES(4,'Cengage Learning');
INSERT INTO "Publisher" VALUES(5,'Elsevier');
CREATE TABLE Subjects (
    id INTEGER PRIMARY KEY,
    subjectName TEXT UNIQUE
);
INSERT INTO "Subjects" VALUES(1,'Math');
INSERT INTO "Subjects" VALUES(2,'Science');
INSERT INTO "Subjects" VALUES(3,'English');
INSERT INTO "Subjects" VALUES(4,'History');
INSERT INTO "Subjects" VALUES(5,'Arts');
CREATE TABLE Teachers (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    subjectId INTEGER,
    FOREIGN KEY (subjectId) REFERENCES Subjects (id)
);
INSERT INTO "Teachers" VALUES(1,'Michael','Williams',1);
INSERT INTO "Teachers" VALUES(2,'Emma','Jones',2);
INSERT INTO "Teachers" VALUES(3,'David','Martin',3);
INSERT INTO "Teachers" VALUES(4,'Laura','Thompson',4);
INSERT INTO "Teachers" VALUES(5,'Daniel','Green',5);
CREATE TABLE Textbooks (
    id INTEGER PRIMARY KEY,
    title TEXT,
    publisherId INTEGER,
    author TEXT,
    price REAL ,
    FOREIGN KEY (publisherId) REFERENCES Publisher (id)
);
INSERT INTO "Textbooks" VALUES(1,'Calculus',1,'Jones',29.99);
INSERT INTO "Textbooks" VALUES(2,'Physics',1,'Smith',34.99);
INSERT INTO "Textbooks" VALUES(3,'Literature',2,'Doe',28.99);
INSERT INTO "Textbooks" VALUES(4,'History',2,'Johnson',30.99);
INSERT INTO "Textbooks" VALUES(5,'Biology',3,'Brown',32.99);
COMMIT;
