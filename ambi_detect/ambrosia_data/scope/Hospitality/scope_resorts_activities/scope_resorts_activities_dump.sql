BEGIN TRANSACTION;
CREATE TABLE Activities(
    activityID INTEGER PRIMARY KEY AUTOINCREMENT,
    activityName TEXT);
INSERT INTO "Activities" VALUES(1,'Swimming Pool');
INSERT INTO "Activities" VALUES(2,'Golf Course');
INSERT INTO "Activities" VALUES(3,'Fitness Center');
INSERT INTO "Activities" VALUES(4,'Restaurant');
INSERT INTO "Activities" VALUES(5,'Kids Club');
INSERT INTO "Activities" VALUES(6,'Beach Volleyball');
INSERT INTO "Activities" VALUES(7,'Scuba Diving');
INSERT INTO "Activities" VALUES(8,'Yoga Classes');
INSERT INTO "Activities" VALUES(9,'Spa Services');
INSERT INTO "Activities" VALUES(10,'Watersports');
CREATE TABLE Resorts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Resorts" VALUES(1,'Paradise Beach','Maldives');
INSERT INTO "Resorts" VALUES(2,'Mountain View Lodge','Colorado');
INSERT INTO "Resorts" VALUES(3,'Tropical Oasis','Hawaii');
INSERT INTO "Resorts" VALUES(4,'Crystal Waters','Australia');
INSERT INTO "Resorts" VALUES(5,'Desert Shadows','Nevada');
CREATE TABLE Resorts_Activities(
    resortID INTEGER,
    activityID INTEGER,
    FOREIGN KEY(resortID) REFERENCES Resorts(id),
    FOREIGN KEY(activityID) REFERENCES Activities(activityID));
INSERT INTO "Resorts_Activities" VALUES(1,1);
INSERT INTO "Resorts_Activities" VALUES(2,1);
INSERT INTO "Resorts_Activities" VALUES(3,2);
INSERT INTO "Resorts_Activities" VALUES(4,2);
INSERT INTO "Resorts_Activities" VALUES(5,3);
INSERT INTO "Resorts_Activities" VALUES(1,6);
INSERT INTO "Resorts_Activities" VALUES(2,7);
INSERT INTO "Resorts_Activities" VALUES(3,8);
INSERT INTO "Resorts_Activities" VALUES(4,9);
INSERT INTO "Resorts_Activities" VALUES(5,10);
INSERT INTO "Resorts_Activities" VALUES(3,1);
INSERT INTO "Resorts_Activities" VALUES(4,1);
INSERT INTO "Resorts_Activities" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Resorts',5);
INSERT INTO "sqlite_sequence" VALUES('Activities',10);
COMMIT;
