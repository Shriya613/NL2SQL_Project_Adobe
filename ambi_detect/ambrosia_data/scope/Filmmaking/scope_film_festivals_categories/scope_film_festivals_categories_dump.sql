BEGIN TRANSACTION;
CREATE TABLE Categories(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    categoryName TEXT
);
INSERT INTO "Categories" VALUES(1,'Short_Film');
INSERT INTO "Categories" VALUES(2,'Feature_Film');
INSERT INTO "Categories" VALUES(3,'Documentary');
INSERT INTO "Categories" VALUES(4,'Animation');
INSERT INTO "Categories" VALUES(5,'Experimental');
CREATE TABLE Film_Festival_Categories(
    festivalId INTEGER,
    categoryId INTEGER,
    FOREIGN KEY (festivalId) REFERENCES Film_Festivals(id),
    FOREIGN KEY (categoryId) REFERENCES Categories(id)
);
INSERT INTO "Film_Festival_Categories" VALUES(1,1);
INSERT INTO "Film_Festival_Categories" VALUES(1,2);
INSERT INTO "Film_Festival_Categories" VALUES(2,4);
INSERT INTO "Film_Festival_Categories" VALUES(2,3);
INSERT INTO "Film_Festival_Categories" VALUES(3,5);
INSERT INTO "Film_Festival_Categories" VALUES(2,1);
INSERT INTO "Film_Festival_Categories" VALUES(3,1);
INSERT INTO "Film_Festival_Categories" VALUES(4,1);
INSERT INTO "Film_Festival_Categories" VALUES(5,1);
CREATE TABLE Film_Festival_Entries(
    festivalId INTEGER,
    filmId INTEGER,
    entryDate DATE,
    FOREIGN KEY (festivalId) REFERENCES Film_Festivals(id),
    FOREIGN KEY (filmId) REFERENCES Films(id)
);
INSERT INTO "Film_Festival_Entries" VALUES(1,1,'2021-07-01');
INSERT INTO "Film_Festival_Entries" VALUES(1,2,'2021-08-01');
INSERT INTO "Film_Festival_Entries" VALUES(2,3,'2022-06-01');
INSERT INTO "Film_Festival_Entries" VALUES(2,4,'2022-07-01');
INSERT INTO "Film_Festival_Entries" VALUES(3,5,'2023-05-01');
CREATE TABLE Film_Festival_Winners(
    festivalId INTEGER,
    awardType TEXT,
    winningFilmId INTEGER,
    winYear INTEGER ,
    FOREIGN KEY (festivalId) REFERENCES Film_Festivals(id),
    FOREIGN KEY (winningFilmId) REFERENCES Films(id)
);
INSERT INTO "Film_Festival_Winners" VALUES(1,'Best Short Film',1,2021);
INSERT INTO "Film_Festival_Winners" VALUES(1,'Best Feature Film',2,2021);
INSERT INTO "Film_Festival_Winners" VALUES(2,'Best Animation',3,2022);
INSERT INTO "Film_Festival_Winners" VALUES(2,'Best Documentary',4,2022);
INSERT INTO "Film_Festival_Winners" VALUES(3,'Best Experimental',5,2023);
CREATE TABLE Film_Festivals(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Film_Festivals" VALUES(1,'Festival A','Location X');
INSERT INTO "Film_Festivals" VALUES(2,'Festival B','Location Y');
INSERT INTO "Film_Festivals" VALUES(3,'Festival C','Location Z');
INSERT INTO "Film_Festivals" VALUES(4,'Festival D','Location W');
INSERT INTO "Film_Festivals" VALUES(5,'Festival E','Location V');
CREATE TABLE Films(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    director TEXT,
    yearReleased INTEGER 
);
INSERT INTO "Films" VALUES(1,'Movie A','Director X',2020);
INSERT INTO "Films" VALUES(2,'Movie B','Director Y',2021);
INSERT INTO "Films" VALUES(3,'Movie C','Director Z',2019);
INSERT INTO "Films" VALUES(4,'Movie D','Director W',2022);
INSERT INTO "Films" VALUES(5,'Movie E','Director V',2023);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Films',5);
INSERT INTO "sqlite_sequence" VALUES('Categories',5);
INSERT INTO "sqlite_sequence" VALUES('Film_Festivals',5);
COMMIT;
