BEGIN TRANSACTION;
CREATE TABLE Art_Galleries (
    GalleryID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT
);
INSERT INTO "Art_Galleries" VALUES(1,'Gallery A','New York');
INSERT INTO "Art_Galleries" VALUES(2,'Gallery B','Los Angeles');
INSERT INTO "Art_Galleries" VALUES(3,'Gallery C','Chicago');
INSERT INTO "Art_Galleries" VALUES(4,'Gallery D','Miami');
INSERT INTO "Art_Galleries" VALUES(5,'Gallery E','San Francisco');
CREATE TABLE Art_Galleries_Exhibitions (
    GalleryID INTEGER,
    ExhibitionID INTEGER,
    FOREIGN KEY(GalleryID) REFERENCES Art_Galleries(GalleryID),
    FOREIGN KEY(ExhibitionID) REFERENCES Exhibitions(ExhibitionID),
    PRIMARY KEY(GalleryID, ExhibitionID)
);
INSERT INTO "Art_Galleries_Exhibitions" VALUES(1,1);
INSERT INTO "Art_Galleries_Exhibitions" VALUES(2,1);
INSERT INTO "Art_Galleries_Exhibitions" VALUES(3,1);
INSERT INTO "Art_Galleries_Exhibitions" VALUES(4,1);
INSERT INTO "Art_Galleries_Exhibitions" VALUES(5,1);
INSERT INTO "Art_Galleries_Exhibitions" VALUES(5,5);
CREATE TABLE Exhibitions (
    ExhibitionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT,
    Description TEXT
);
INSERT INTO "Exhibitions" VALUES(1,'Painting_Exhibitions','An exhibition showcasing famous paintings from around the world.');
INSERT INTO "Exhibitions" VALUES(2,'Sculpture_Exhibitions','A collection of modern sculptures from various artists.');
INSERT INTO "Exhibitions" VALUES(3,'Photography_Exhibitions','A display of stunning photography from renowned photographers.');
INSERT INTO "Exhibitions" VALUES(4,'Digital_Art_Exhibitions','A showcase of digital art created using cutting-edge technology.');
INSERT INTO "Exhibitions" VALUES(5,'Installation_Art_Exhibitions','Immersive installations that engage all senses and create unique experiences.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Art_Galleries',5);
INSERT INTO "sqlite_sequence" VALUES('Exhibitions',5);
COMMIT;
