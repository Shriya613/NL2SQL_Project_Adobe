BEGIN TRANSACTION;
CREATE TABLE OnlineResources (
    resourceId INTEGER PRIMARY KEY,
    url TEXT,
    title TEXT);
INSERT INTO "OnlineResources" VALUES(1,'https://www.mathopenref.com/','Math Open Ref');
INSERT INTO "OnlineResources" VALUES(2,'https://www.english-test.net/','English Test Net');
INSERT INTO "OnlineResources" VALUES(3,'https://www.historyworld.org/','History World');
INSERT INTO "OnlineResources" VALUES(4,'https://www.physicsclassroom.com/','Physics Classroom');
INSERT INTO "OnlineResources" VALUES(5,'https://www.chemguide.co.uk/','Chem Guide');
CREATE TABLE "TestPreparations" (
    preparationId INTEGER PRIMARY KEY,
    topic TEXT,
    resourcesId INTEGER,
    "servicesId" INTEGER,
    FOREIGN KEY(resourcesId) REFERENCES OnlineResources(resourceId),
    FOREIGN KEY("servicesId") REFERENCES TutoringServices(serviceId));
INSERT INTO "TestPreparations" VALUES(1,'Algebra',1,1);
INSERT INTO "TestPreparations" VALUES(2,'Grammar',2,2);
INSERT INTO "TestPreparations" VALUES(3,'Ancient History',3,3);
INSERT INTO "TestPreparations" VALUES(4,'Electricity and Magnetism',4,4);
INSERT INTO "TestPreparations" VALUES(5,'Organic Chemistry',5,5);
CREATE TABLE "TutoringServices" (
    serviceId INTEGER PRIMARY KEY,
    tutorId INTEGER,
    date TIMESTAMP,
    duration INTEGER,
    FOREIGN KEY(tutorId) REFERENCES Students(id));
INSERT INTO "TutoringServices" VALUES(1,1,'2022-12-19 18:00:00',60);
INSERT INTO "TutoringServices" VALUES(2,2,'2022-12-20 17:30:00',45);
INSERT INTO "TutoringServices" VALUES(3,3,'2022-12-21 16:00:00',60);
INSERT INTO "TutoringServices" VALUES(4,4,'2022-12-22 15:30:00',60);
INSERT INTO "TutoringServices" VALUES(5,5,'2022-12-23 17:00:00',75);
CREATE TABLE "Tutors" (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT);
INSERT INTO "Tutors" VALUES(1,'John','Doe');
INSERT INTO "Tutors" VALUES(2,'Jane','Smith');
INSERT INTO "Tutors" VALUES(3,'Michael','Brown');
INSERT INTO "Tutors" VALUES(4,'Emily','White');
INSERT INTO "Tutors" VALUES(5,'David','Green');
COMMIT;
