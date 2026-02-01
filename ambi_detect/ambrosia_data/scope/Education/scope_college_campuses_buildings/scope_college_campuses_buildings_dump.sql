BEGIN TRANSACTION;
CREATE TABLE Buildings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    description TEXT);
INSERT INTO "Buildings" VALUES(1,'Library','A place where books and resources are stored.');
INSERT INTO "Buildings" VALUES(2,'Administration Building','The main office building on campus.');
INSERT INTO "Buildings" VALUES(3,'Lecture Hall','Large rooms used for lectures and presentations.');
INSERT INTO "Buildings" VALUES(4,'Dormitory','Residential living space for students.');
INSERT INTO "Buildings" VALUES(5,'Gymnasium','Facility for sports activities.');
CREATE TABLE College_Campuses(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "College_Campuses" VALUES(1,'Harvard University','Cambridge, MA');
INSERT INTO "College_Campuses" VALUES(2,'Stanford University','Stanford, CA');
INSERT INTO "College_Campuses" VALUES(3,'University of California Berkeley','Berkeley, CA');
INSERT INTO "College_Campuses" VALUES(4,'Massachusetts Institute of Technology','Cambridge, MA');
INSERT INTO "College_Campuses" VALUES(5,'California Institute of Technology','Pasadena, CA');
CREATE TABLE College_Campuses_Buildings(
    campusId INTEGER REFERENCES College_Campuses(id) ON DELETE SET NULL,
    buildingId INTEGER REFERENCES Buildings(id) ON DELETE SET NULL,
    FOREIGN KEY (campusId) REFERENCES College_Campuses(id) ON DELETE SET NULL,
    FOREIGN KEY (buildingId) REFERENCES Buildings(id) ON DELETE SET NULL,
    PRIMARY KEY (campusId, buildingId));
INSERT INTO "College_Campuses_Buildings" VALUES(1,1);
INSERT INTO "College_Campuses_Buildings" VALUES(1,2);
INSERT INTO "College_Campuses_Buildings" VALUES(2,1);
INSERT INTO "College_Campuses_Buildings" VALUES(2,3);
INSERT INTO "College_Campuses_Buildings" VALUES(3,1);
INSERT INTO "College_Campuses_Buildings" VALUES(4,1);
INSERT INTO "College_Campuses_Buildings" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('College_Campuses',5);
INSERT INTO "sqlite_sequence" VALUES('Buildings',5);
COMMIT;
