BEGIN TRANSACTION;
CREATE TABLE Preservation_Societies(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT
);
INSERT INTO "Preservation_Societies" VALUES(1,'Liberty Hill Society','123 Main St');
INSERT INTO "Preservation_Societies" VALUES(2,'Heritage Park Society','456 Elm St');
INSERT INTO "Preservation_Societies" VALUES(3,'Old Town Village Society','789 Oak St');
INSERT INTO "Preservation_Societies" VALUES(4,'Woodland Heights Society','001 Pine St');
INSERT INTO "Preservation_Societies" VALUES(5,'Riverbend Community Society','135 Maple Ave');
CREATE TABLE Preservation_Societies_Projects(
    Societies_ID INTEGER,
    Projects_ID INTEGER,
    StartDate DATE,
    EndDate DATE,
    FOREIGN KEY (Societies_ID) REFERENCES Preservation_Societies(ID),
    FOREIGN KEY (Projects_ID) REFERENCES Projects(ID),
    PRIMARY KEY (Societies_ID, Projects_ID)
);
INSERT INTO "Preservation_Societies_Projects" VALUES(1,1,'2022-01-01','2022-12-31');
INSERT INTO "Preservation_Societies_Projects" VALUES(1,2,'2022-01-01','2022-12-31');
INSERT INTO "Preservation_Societies_Projects" VALUES(2,1,'2022-01-01','2022-12-31');
INSERT INTO "Preservation_Societies_Projects" VALUES(2,2,'2022-01-01','2022-12-31');
INSERT INTO "Preservation_Societies_Projects" VALUES(3,1,'2022-01-01','2022-12-31');
INSERT INTO "Preservation_Societies_Projects" VALUES(3,3,'2022-01-01','2022-12-31');
INSERT INTO "Preservation_Societies_Projects" VALUES(4,2,'2022-01-01','2022-12-31');
INSERT INTO "Preservation_Societies_Projects" VALUES(4,4,'2022-01-01','2022-12-31');
INSERT INTO "Preservation_Societies_Projects" VALUES(5,3,'2022-01-01','2022-12-31');
INSERT INTO "Preservation_Societies_Projects" VALUES(5,5,'2022-01-01','2022-12-31');
INSERT INTO "Preservation_Societies_Projects" VALUES(4,1,'2022-01-01','2022-12-31');
INSERT INTO "Preservation_Societies_Projects" VALUES(5,1,'2022-01-01','2022-12-31');
CREATE TABLE Projects(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT ,
    Description TEXT,
    Cost REAL
);
INSERT INTO "Projects" VALUES(1,'Restoration','Restoring historical buildings in the neighborhood.',250000.0);
INSERT INTO "Projects" VALUES(2,'Improvement','Upgrading infrastructure and amenities.',150000.0);
INSERT INTO "Projects" VALUES(3,'Maintenance','Regular upkeep of properties.',50000.0);
INSERT INTO "Projects" VALUES(4,'Development','Constructing new houses while preserving the architectural style.',350000.0);
INSERT INTO "Projects" VALUES(5,'Landscaping','Creating green spaces and gardens around the community.',75000.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Preservation_Societies',5);
INSERT INTO "sqlite_sequence" VALUES('Projects',5);
COMMIT;
