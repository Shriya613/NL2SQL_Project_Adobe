BEGIN TRANSACTION;
CREATE TABLE Images(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT,
    description TEXT);
INSERT INTO "Images" VALUES(1,'Candid_Shots','A candid shot of a protestor');
INSERT INTO "Images" VALUES(2,'Building_Collapse','Ruins of a building after a natural disaster');
INSERT INTO "Images" VALUES(3,'Impoverished_Family','A family struggling during an economic crisis');
INSERT INTO "Images" VALUES(4,'Portrait of a Politician','Famous politician is giving a speech.');
CREATE TABLE Photographers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birth_date DATE);
INSERT INTO "Photographers" VALUES(1,'John Doe','1980-01-01');
INSERT INTO "Photographers" VALUES(2,'Jane Smith','1985-02-20');
INSERT INTO "Photographers" VALUES(3,'Michael Brown','1988-03-25');
CREATE TABLE Photojournalism_Pieces(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    date DATE);
INSERT INTO "Photojournalism_Pieces" VALUES(1,'Revolutionary Protest','2021-07-08');
INSERT INTO "Photojournalism_Pieces" VALUES(2,'Natural Disaster Aftermath','2021-06-30');
INSERT INTO "Photojournalism_Pieces" VALUES(3,'Economic Crisis Impact','2021-05-15');
CREATE TABLE Photojournalism_Pieces_Images(
    photojournalism_piece_id INTEGER,
    image_id INTEGER,
    FOREIGN KEY (photojournalism_piece_id) REFERENCES Photojournalism_Pieces(id),
    FOREIGN KEY (image_id) REFERENCES Images(id));
INSERT INTO "Photojournalism_Pieces_Images" VALUES(1,4);
INSERT INTO "Photojournalism_Pieces_Images" VALUES(1,1);
INSERT INTO "Photojournalism_Pieces_Images" VALUES(2,2);
INSERT INTO "Photojournalism_Pieces_Images" VALUES(2,4);
INSERT INTO "Photojournalism_Pieces_Images" VALUES(3,4);
INSERT INTO "Photojournalism_Pieces_Images" VALUES(3,3);
CREATE TABLE Photojournalism_Pieces_Photographer(
    photojournalism_piece_id INTEGER,
    photographer_id INTEGER,
    FOREIGN KEY (photojournalism_piece_id) REFERENCES Photojournalism_Pieces(id),
    FOREIGN KEY (photographer_id) REFERENCES Photographers(id));
INSERT INTO "Photojournalism_Pieces_Photographer" VALUES(1,1);
INSERT INTO "Photojournalism_Pieces_Photographer" VALUES(1,2);
INSERT INTO "Photojournalism_Pieces_Photographer" VALUES(2,1);
INSERT INTO "Photojournalism_Pieces_Photographer" VALUES(2,3);
INSERT INTO "Photojournalism_Pieces_Photographer" VALUES(3,2);
INSERT INTO "Photojournalism_Pieces_Photographer" VALUES(3,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Photojournalism_Pieces',3);
INSERT INTO "sqlite_sequence" VALUES('Images',4);
INSERT INTO "sqlite_sequence" VALUES('Photographers',3);
COMMIT;
