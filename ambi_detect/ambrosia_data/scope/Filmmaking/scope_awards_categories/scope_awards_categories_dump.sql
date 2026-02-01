BEGIN TRANSACTION;
CREATE TABLE Awards(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    awardName TEXT,
    year INT
);
INSERT INTO "Awards" VALUES(1,'Academy Award',2019);
INSERT INTO "Awards" VALUES(2,'Golden Globe',2018);
INSERT INTO "Awards" VALUES(3,'BAFTA',2017);
INSERT INTO "Awards" VALUES(4,'Palme d''Or',2016);
INSERT INTO "Awards" VALUES(5,'Emmy',2015);
CREATE TABLE Awards_Categories(
    awardsID INTEGER,
    categoriesID INTEGER,
    FOREIGN KEY(awardsID) REFERENCES Awards(id),
    FOREIGN KEY(categoriesID) REFERENCES Categories(categoryId),
    PRIMARY KEY(awardsID, categoriesID)
);
INSERT INTO "Awards_Categories" VALUES(2,1);
INSERT INTO "Awards_Categories" VALUES(3,1);
INSERT INTO "Awards_Categories" VALUES(4,1);
INSERT INTO "Awards_Categories" VALUES(5,1);
INSERT INTO "Awards_Categories" VALUES(1,2);
INSERT INTO "Awards_Categories" VALUES(3,2);
INSERT INTO "Awards_Categories" VALUES(4,2);
INSERT INTO "Awards_Categories" VALUES(5,2);
INSERT INTO "Awards_Categories" VALUES(1,3);
INSERT INTO "Awards_Categories" VALUES(2,3);
INSERT INTO "Awards_Categories" VALUES(4,3);
INSERT INTO "Awards_Categories" VALUES(5,3);
INSERT INTO "Awards_Categories" VALUES(1,4);
INSERT INTO "Awards_Categories" VALUES(2,4);
INSERT INTO "Awards_Categories" VALUES(3,4);
INSERT INTO "Awards_Categories" VALUES(5,4);
INSERT INTO "Awards_Categories" VALUES(1,5);
INSERT INTO "Awards_Categories" VALUES(2,5);
INSERT INTO "Awards_Categories" VALUES(3,5);
INSERT INTO "Awards_Categories" VALUES(4,5);
INSERT INTO "Awards_Categories" VALUES(5,5);
INSERT INTO "Awards_Categories" VALUES(1,1);
INSERT INTO "Awards_Categories" VALUES(2,2);
INSERT INTO "Awards_Categories" VALUES(3,3);
INSERT INTO "Awards_Categories" VALUES(4,4);
CREATE TABLE Categories(
    categoryId INTEGER PRIMARY KEY AUTOINCREMENT,
    categoryName TEXT
);
INSERT INTO "Categories" VALUES(1,'Best Actor');
INSERT INTO "Categories" VALUES(2,'Best Director');
INSERT INTO "Categories" VALUES(3,'Best Film');
INSERT INTO "Categories" VALUES(4,'Best Screenplay');
INSERT INTO "Categories" VALUES(5,'Best Cinematography');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Awards',5);
INSERT INTO "sqlite_sequence" VALUES('Categories',5);
COMMIT;
