BEGIN TRANSACTION;
CREATE TABLE Aquaculture_Facility (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Aquaculture_Facility" VALUES(1,'Facility A','Location A');
INSERT INTO "Aquaculture_Facility" VALUES(2,'Facility B','Location B');
INSERT INTO "Aquaculture_Facility" VALUES(3,'Facility C','Location C');
INSERT INTO "Aquaculture_Facility" VALUES(4,'Facility D','Location D');
INSERT INTO "Aquaculture_Facility" VALUES(5,'Facility E','Location E');
CREATE TABLE Aquaculture_Facility_Fish_Species (
    aquaculture_facility_id INTEGER,
    fish_species_id INTEGER,
    FOREIGN KEY (aquaculture_facility_id) REFERENCES Aquaculture_Facility(id),
    FOREIGN KEY (fish_species_id) REFERENCES Fish_Species(id),
    PRIMARY KEY (aquaculture_facility_id, fish_species_id)
);
INSERT INTO "Aquaculture_Facility_Fish_Species" VALUES(1,1);
INSERT INTO "Aquaculture_Facility_Fish_Species" VALUES(2,1);
INSERT INTO "Aquaculture_Facility_Fish_Species" VALUES(3,1);
INSERT INTO "Aquaculture_Facility_Fish_Species" VALUES(4,1);
INSERT INTO "Aquaculture_Facility_Fish_Species" VALUES(5,1);
INSERT INTO "Aquaculture_Facility_Fish_Species" VALUES(5,4);
CREATE TABLE Fish_Species (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    species_name TEXT
);
INSERT INTO "Fish_Species" VALUES(1,'Salmon');
INSERT INTO "Fish_Species" VALUES(2,'Catfish');
INSERT INTO "Fish_Species" VALUES(3,'Tilapia');
INSERT INTO "Fish_Species" VALUES(4,'Trout');
INSERT INTO "Fish_Species" VALUES(5,'Bass');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Aquaculture_Facility',5);
INSERT INTO "sqlite_sequence" VALUES('Fish_Species',5);
COMMIT;
