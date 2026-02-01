BEGIN TRANSACTION;
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT
);
INSERT INTO "Authors" VALUES(1,'Dr. Emily Thompson');
INSERT INTO "Authors" VALUES(2,'Jane Doe');
INSERT INTO "Authors" VALUES(3,'John Smith');
INSERT INTO "Authors" VALUES(4,'Emma Watson');
INSERT INTO "Authors" VALUES(5,'George Orwell');
CREATE TABLE Books_Sales(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    BookID INTEGER,
    Quantity INTEGER,
    SaleDate DATE,
    Revenue REAL,
    FOREIGN KEY(BookID) REFERENCES Books(id)
);
INSERT INTO "Books_Sales" VALUES(1,1,10,'2023-01-01',1000.0);
INSERT INTO "Books_Sales" VALUES(2,2,5,'2023-01-02',500.0);
CREATE TABLE Movie_Ratings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieID INTEGER,
    Rating TEXT,
    ReviewerID INTEGER,
    DateRated DATE,
    FOREIGN KEY(MovieID) REFERENCES Movies(id),
    FOREIGN KEY(ReviewerID) REFERENCES Reviewers(id)
);
INSERT INTO "Movie_Ratings" VALUES(1,1,'Awesome',3,'2023-01-01');
INSERT INTO "Movie_Ratings" VALUES(2,2,'Good',4,'2023-01-02');
CREATE TABLE Written_Works(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    AuthorID INTEGER,
    Type TEXT,
    Title TEXT,
    DatePublished DATE,
    FOREIGN KEY(AuthorID) REFERENCES Authors(id)
);
INSERT INTO "Written_Works" VALUES(1,1,'Essay','My Favorite Essay','2023-01-01');
INSERT INTO "Written_Works" VALUES(2,2,'Essay','Some Other Essay','2023-01-02');
INSERT INTO "Written_Works" VALUES(3,1,'Letter','Important Letter','2023-01-03');
INSERT INTO "Written_Works" VALUES(4,2,'Letter','Another Important Letter','2023-01-04');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Written_Works',4);
INSERT INTO "sqlite_sequence" VALUES('Movie_Ratings',2);
INSERT INTO "sqlite_sequence" VALUES('Books_Sales',2);
COMMIT;
