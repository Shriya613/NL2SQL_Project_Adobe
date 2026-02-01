BEGIN TRANSACTION;
CREATE TABLE ArtType (
    id INTEGER PRIMARY KEY,
    TypeName TEXT UNIQUE
);
INSERT INTO "ArtType" VALUES(1,'Photography');
INSERT INTO "ArtType" VALUES(2,'Painting');
CREATE TABLE Artist (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    BirthYear INT,
    DeathYear INT
);
INSERT INTO "Artist" VALUES(1,'Vincent van Gogh',1853,1890);
INSERT INTO "Artist" VALUES(2,'Claude Monet',1840,1926);
CREATE TABLE ArtworkInExhibition (
    id INTEGER PRIMARY KEY,
    ExhibitId INTEGER,
    VisualArtId INTEGER,
    FOREIGN KEY(ExhibitId) REFERENCES Exhibit(id),
    FOREIGN KEY(VisualArtId) REFERENCES VisualArt(id)
);
INSERT INTO "ArtworkInExhibition" VALUES(1,1,1);
INSERT INTO "ArtworkInExhibition" VALUES(2,1,2);
INSERT INTO "ArtworkInExhibition" VALUES(3,2,3);
INSERT INTO "ArtworkInExhibition" VALUES(4,2,4);
CREATE TABLE Exhibit (
    id INTEGER PRIMARY KEY,
    GalleryId INTEGER,
    StartDate DATE,
    EndDate DATE,
    FOREIGN KEY(GalleryId) REFERENCES Gallery(id)
);
INSERT INTO "Exhibit" VALUES(1,1,'2023-01-01','2023-01-31');
INSERT INTO "Exhibit" VALUES(2,2,'2023-02-01','2023-02-28');
CREATE TABLE Gallery (
    id INTEGER PRIMARY KEY,
    Name TEXT UNIQUE
);
INSERT INTO "Gallery" VALUES(1,'MoMA');
INSERT INTO "Gallery" VALUES(2,'National Gallery of Art');
CREATE TABLE VisualArt (
    id INTEGER PRIMARY KEY,
    ArtTypeId INTEGER,
    Title TEXT,
    Description TEXT,
    FOREIGN KEY(ArtTypeId) REFERENCES ArtType(id)
);
INSERT INTO "VisualArt" VALUES(1,1,'The Moon','A beautiful photograph of the moon.');
INSERT INTO "VisualArt" VALUES(2,2,'Starry Night','A famous painting by Van Gogh.');
INSERT INTO "VisualArt" VALUES(3,1,'Nature Walk','Another photography masterpiece.');
INSERT INTO "VisualArt" VALUES(4,2,'Sunset at Sea','A breathtaking sunset captured on canvas.');
COMMIT;
