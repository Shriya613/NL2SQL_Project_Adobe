BEGIN TRANSACTION;
CREATE TABLE Graphic_Designers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    experience INTEGER
);
INSERT INTO "Graphic_Designers" VALUES(1,'John Doe',3);
INSERT INTO "Graphic_Designers" VALUES(2,'Jane Smith',2);
INSERT INTO "Graphic_Designers" VALUES(3,'Mike Johnson',5);
INSERT INTO "Graphic_Designers" VALUES(4,'Emily Davis',10);
INSERT INTO "Graphic_Designers" VALUES(5,'David Brown',7);
CREATE TABLE Graphic_Designers_Programs(
    graphicDesignerId INTEGER,
    programId INTEGER,
    FOREIGN KEY(graphicDesignerId) REFERENCES Graphic_Designers(id),
    FOREIGN KEY(programId) REFERENCES Programs(id)
);
INSERT INTO "Graphic_Designers_Programs" VALUES(1,1);
INSERT INTO "Graphic_Designers_Programs" VALUES(2,1);
INSERT INTO "Graphic_Designers_Programs" VALUES(3,1);
INSERT INTO "Graphic_Designers_Programs" VALUES(4,1);
INSERT INTO "Graphic_Designers_Programs" VALUES(5,1);
INSERT INTO "Graphic_Designers_Programs" VALUES(5,4);
CREATE TABLE Programs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    programName TEXT
);
INSERT INTO "Programs" VALUES(1,'Adobe Illustrator');
INSERT INTO "Programs" VALUES(2,'Photoshop');
INSERT INTO "Programs" VALUES(3,'InDesign');
INSERT INTO "Programs" VALUES(4,'Sketch');
INSERT INTO "Programs" VALUES(5,'Figma');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Graphic_Designers',5);
INSERT INTO "sqlite_sequence" VALUES('Programs',5);
COMMIT;
