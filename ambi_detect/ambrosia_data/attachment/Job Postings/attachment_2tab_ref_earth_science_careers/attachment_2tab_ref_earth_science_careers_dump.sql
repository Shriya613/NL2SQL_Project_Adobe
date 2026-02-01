BEGIN TRANSACTION;
CREATE TABLE Chemist (
    id INTEGER PRIMARY KEY,
    university_id INTEGER,
    fieldOfStudy TEXT, FullName TEXT,
    FOREIGN KEY(university_id) REFERENCES University(id)
);
INSERT INTO "Chemist" VALUES(1,1,'Organic Chemistry','John Doe');
INSERT INTO "Chemist" VALUES(2,1,'Inorganic Chemistry','Jane Smith');
INSERT INTO "Chemist" VALUES(3,2,'Physical Chemistry','Alice Brown');
INSERT INTO "Chemist" VALUES(4,2,'Analytical Chemistry','Bob White');
CREATE TABLE Geoscientist (
    id INTEGER PRIMARY KEY,
    university_id INTEGER,
    fieldOfStudy TEXT, FullName TEXT,
    FOREIGN KEY(university_id) REFERENCES University(id)
);
INSERT INTO "Geoscientist" VALUES(1,1,'Geology','Tom Jackson');
INSERT INTO "Geoscientist" VALUES(2,1,'Seismology','Emma Nelson');
INSERT INTO "Geoscientist" VALUES(3,2,'Paleontology','Mike Garcia');
INSERT INTO "Geoscientist" VALUES(4,2,'Oceanography','Lisa Hill');
CREATE TABLE University (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);
INSERT INTO "University" VALUES(1,'MIT');
INSERT INTO "University" VALUES(2,'Stanford');
COMMIT;
