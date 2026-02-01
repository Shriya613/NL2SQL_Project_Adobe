BEGIN TRANSACTION;
CREATE TABLE Categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT
);
INSERT INTO "Categories" VALUES(1,'Documentary_Films');
INSERT INTO "Categories" VALUES(2,'Narrative_Films');
INSERT INTO "Categories" VALUES(3,'Experimental_Films');
INSERT INTO "Categories" VALUES(4,'Animation_Films');
INSERT INTO "Categories" VALUES(5,'Short_Films');
CREATE TABLE Film_Festivals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Film_Festivals" VALUES(1,'New York International Documentary Festival','New York City');
INSERT INTO "Film_Festivals" VALUES(2,'Los Angeles Short Film Festival','Los Angeles');
INSERT INTO "Film_Festivals" VALUES(3,'Austin Experimental Film Festival','Austin');
INSERT INTO "Film_Festivals" VALUES(4,'International Animation Film Festival','London');
INSERT INTO "Film_Festivals" VALUES(5,'World Narrative Film Festival','Toronto');
CREATE TABLE Film_Festivals_Categories (
    film_festival_id INTEGER,
    category_id INTEGER,
    FOREIGN KEY(film_festival_id) REFERENCES Film_Festivals(id),
    FOREIGN KEY(category_id) REFERENCES Categories(id)
);
INSERT INTO "Film_Festivals_Categories" VALUES(1,1);
INSERT INTO "Film_Festivals_Categories" VALUES(2,1);
INSERT INTO "Film_Festivals_Categories" VALUES(3,1);
INSERT INTO "Film_Festivals_Categories" VALUES(4,1);
INSERT INTO "Film_Festivals_Categories" VALUES(5,1);
INSERT INTO "Film_Festivals_Categories" VALUES(5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Categories',5);
INSERT INTO "sqlite_sequence" VALUES('Film_Festivals',5);
COMMIT;
