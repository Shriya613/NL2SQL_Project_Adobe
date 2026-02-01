BEGIN TRANSACTION;
CREATE TABLE Field_Trips(
    tripID INTEGER PRIMARY KEY,
    fieldTripName TEXT,
    date DATE);
INSERT INTO "Field_Trips" VALUES(1,'Educational Trip','2021-09-01');
INSERT INTO "Field_Trips" VALUES(2,'Scientific Excursion','2021-08-30');
INSERT INTO "Field_Trips" VALUES(3,'Nature Adventure','2021-07-25');
INSERT INTO "Field_Trips" VALUES(4,'Historical Tour','2021-06-10');
INSERT INTO "Field_Trips" VALUES(5,'Art Exhibition','2021-05-01');
CREATE TABLE Field_Trips_Stops(
    tripID INTEGER REFERENCES Field_Trips(tripID),
    stopID INTEGER REFERENCES Stops(stopID));
INSERT INTO "Field_Trips_Stops" VALUES(1,1);
INSERT INTO "Field_Trips_Stops" VALUES(1,2);
INSERT INTO "Field_Trips_Stops" VALUES(5,5);
INSERT INTO "Field_Trips_Stops" VALUES(3,3);
INSERT INTO "Field_Trips_Stops" VALUES(4,4);
INSERT INTO "Field_Trips_Stops" VALUES(2,2);
INSERT INTO "Field_Trips_Stops" VALUES(3,2);
INSERT INTO "Field_Trips_Stops" VALUES(4,2);
INSERT INTO "Field_Trips_Stops" VALUES(5,2);
CREATE TABLE Field_Trips_Students(
    tripID INTEGER REFERENCES Field_Trips(tripID),
    studentID INTEGER REFERENCES Students(studentID));
INSERT INTO "Field_Trips_Students" VALUES(1,1);
INSERT INTO "Field_Trips_Students" VALUES(1,2);
INSERT INTO "Field_Trips_Students" VALUES(2,3);
INSERT INTO "Field_Trips_Students" VALUES(2,4);
INSERT INTO "Field_Trips_Students" VALUES(3,5);
CREATE TABLE Field_Trips_Teachers(
    tripID INTEGER REFERENCES Field_Trips(tripID),
    teacherID INTEGER REFERENCES Teachers(teacherID));
INSERT INTO "Field_Trips_Teachers" VALUES(1,1);
INSERT INTO "Field_Trips_Teachers" VALUES(2,2);
INSERT INTO "Field_Trips_Teachers" VALUES(3,3);
INSERT INTO "Field_Trips_Teachers" VALUES(4,4);
INSERT INTO "Field_Trips_Teachers" VALUES(5,5);
CREATE TABLE Stops(
    stopID INTEGER PRIMARY KEY,
    stopLocation TEXT);
INSERT INTO "Stops" VALUES(1,'Zoo');
INSERT INTO "Stops" VALUES(2,'Museum');
INSERT INTO "Stops" VALUES(3,'Botanical Garden');
INSERT INTO "Stops" VALUES(4,'Planetarium');
INSERT INTO "Stops" VALUES(5,'National Park');
CREATE TABLE Students(
    studentID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT);
INSERT INTO "Students" VALUES(1,'John','Doe');
INSERT INTO "Students" VALUES(2,'Jane','Smith');
INSERT INTO "Students" VALUES(3,'Alice','Brown');
INSERT INTO "Students" VALUES(4,'Tom','Johnson');
INSERT INTO "Students" VALUES(5,'Emma','Wilson');
CREATE TABLE Teachers(
    teacherID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT);
INSERT INTO "Teachers" VALUES(1,'Mr.','White');
INSERT INTO "Teachers" VALUES(2,'Ms.','Black');
INSERT INTO "Teachers" VALUES(3,'Dr.','Green');
INSERT INTO "Teachers" VALUES(4,'Professor','Blue');
INSERT INTO "Teachers" VALUES(5,'Assistant','Red');
COMMIT;
