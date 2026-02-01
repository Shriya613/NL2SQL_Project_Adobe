BEGIN TRANSACTION;
CREATE TABLE Categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    categoryName TEXT
);
INSERT INTO "Categories" VALUES(1,'Lifestyle');
INSERT INTO "Categories" VALUES(2,'Technology');
INSERT INTO "Categories" VALUES(3,'Politics');
INSERT INTO "Categories" VALUES(4,'Travel');
INSERT INTO "Categories" VALUES(5,'Health');
CREATE TABLE Magazines (
    magazineId INTEGER PRIMARY KEY AUTOINCREMENT,
    magazineTitle TEXT,
    publicationYear INTEGER,
    websiteUrl TEXT
);
INSERT INTO "Magazines" VALUES(1,'Vogue',2008,'https://www.vogue.com');
INSERT INTO "Magazines" VALUES(2,'Wired',1993,'https://www.wired.com');
INSERT INTO "Magazines" VALUES(3,'The Economist',1867,'https://www.economist.com');
INSERT INTO "Magazines" VALUES(4,'National Geographic',1888,'https://www.nationalgeographic.com');
INSERT INTO "Magazines" VALUES(5,'Men''s Health',1987,'https://www.menshealth.com');
CREATE TABLE Magazines_Categories (
    magazineId INTEGER,
    categoryId INTEGER,
    FOREIGN KEY (magazineId) REFERENCES Magazines(magazineId),
    FOREIGN KEY (categoryId) REFERENCES Categories(id),
    PRIMARY KEY (magazineId, categoryId)
);
INSERT INTO "Magazines_Categories" VALUES(1,1);
INSERT INTO "Magazines_Categories" VALUES(2,1);
INSERT INTO "Magazines_Categories" VALUES(3,1);
INSERT INTO "Magazines_Categories" VALUES(4,1);
INSERT INTO "Magazines_Categories" VALUES(5,1);
INSERT INTO "Magazines_Categories" VALUES(5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Categories',5);
INSERT INTO "sqlite_sequence" VALUES('Magazines',5);
COMMIT;
