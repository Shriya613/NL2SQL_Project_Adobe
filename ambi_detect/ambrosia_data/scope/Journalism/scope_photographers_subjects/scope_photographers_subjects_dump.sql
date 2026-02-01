BEGIN TRANSACTION;
CREATE TABLE Photographers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Photographers" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Photographers" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Photographers" VALUES(3,'Michael Johnson','michael@example.com');
INSERT INTO "Photographers" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Photographers" VALUES(5,'David Brown','david@example.com');
CREATE TABLE Photographers_Subjects (
    photographer_id INTEGER,
    subject_id INTEGER,
    FOREIGN KEY(photographer_id) REFERENCES Photographers(id),
    FOREIGN KEY(subject_id) REFERENCES Subjects(id),
    PRIMARY KEY(photographer_id, subject_id)
);
INSERT INTO "Photographers_Subjects" VALUES(1,1);
INSERT INTO "Photographers_Subjects" VALUES(2,1);
INSERT INTO "Photographers_Subjects" VALUES(3,1);
INSERT INTO "Photographers_Subjects" VALUES(4,1);
INSERT INTO "Photographers_Subjects" VALUES(5,1);
INSERT INTO "Photographers_Subjects" VALUES(5,5);
CREATE TABLE Subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_name TEXT
);
INSERT INTO "Subjects" VALUES(1,'Portraits');
INSERT INTO "Subjects" VALUES(2,'Landscapes');
INSERT INTO "Subjects" VALUES(3,'Architecture');
INSERT INTO "Subjects" VALUES(4,'Wildlife');
INSERT INTO "Subjects" VALUES(5,'Street Photography');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Photographers',5);
INSERT INTO "sqlite_sequence" VALUES('Subjects',5);
COMMIT;
