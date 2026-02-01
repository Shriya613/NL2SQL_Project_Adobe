BEGIN TRANSACTION;
CREATE TABLE Genres (
    genreID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Genres" VALUES(1,'Blog_Post');
INSERT INTO "Genres" VALUES(2,'Fiction');
INSERT INTO "Genres" VALUES(3,'Non-fiction');
INSERT INTO "Genres" VALUES(4,'Poetry');
INSERT INTO "Genres" VALUES(5,'Essays');
CREATE TABLE Writers (
    writerID INTEGER PRIMARY KEY AUTOINCREMENT,
    fullName TEXT);
INSERT INTO "Writers" VALUES(1,'John Doe');
INSERT INTO "Writers" VALUES(2,'Jane Smith');
INSERT INTO "Writers" VALUES(3,'Michael Johnson');
INSERT INTO "Writers" VALUES(4,'Emily Williams');
INSERT INTO "Writers" VALUES(5,'David Brown');
CREATE TABLE Writers_Genres (
    writerID INTEGER,
    genreID INTEGER,
    FOREIGN KEY (writerID) REFERENCES Writers(writerID),
    FOREIGN KEY (genreID) REFERENCES Genres(genreID),
    PRIMARY KEY (writerID, genreID)
);
INSERT INTO "Writers_Genres" VALUES(1,1);
INSERT INTO "Writers_Genres" VALUES(2,1);
INSERT INTO "Writers_Genres" VALUES(3,1);
INSERT INTO "Writers_Genres" VALUES(4,1);
INSERT INTO "Writers_Genres" VALUES(5,1);
INSERT INTO "Writers_Genres" VALUES(5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Writers',5);
INSERT INTO "sqlite_sequence" VALUES('Genres',5);
COMMIT;
