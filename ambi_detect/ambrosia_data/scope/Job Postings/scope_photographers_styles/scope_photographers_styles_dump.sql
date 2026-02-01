BEGIN TRANSACTION;
CREATE TABLE BookedJobs (
    bookingID INTEGER PRIMARY KEY AUTOINCREMENT,
    photograherId INTEGER REFERENCES Photographers(id),
    jobId INTEGER REFERENCES JobPostings(jobID),
    bookDate DATE,
    paymentAmount DECIMAL(10,2));
CREATE TABLE JobPostings (
    jobID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    deadline DATE);
INSERT INTO "JobPostings" VALUES(1,'Family Portrait Session','A beautiful family portrait session at your home.','2023-08-01');
INSERT INTO "JobPostings" VALUES(2,'Landscape Photography Shoot','Capture stunning landscapes for a travel blog.','2023-07-20');
INSERT INTO "JobPostings" VALUES(3,'Wedding Ceremony Coverage','Document the special day of a couple getting married.','2023-09-01');
INSERT INTO "JobPostings" VALUES(4,'Fashion Photo Shoot','Create a high fashion photo shoot for an upcoming clothing line.','2023-06-15');
INSERT INTO "JobPostings" VALUES(5,'Corporate Event Coverage','Capture the essence of a corporate event for marketing purposes.','2023-05-30');
CREATE TABLE Photographers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT);
INSERT INTO "Photographers" VALUES(1,'John Smith','johnsmith@example.com');
INSERT INTO "Photographers" VALUES(2,'Emily Johnson','emilyjohnson@example.com');
INSERT INTO "Photographers" VALUES(3,'Michael Brown','michaelbrown@example.com');
INSERT INTO "Photographers" VALUES(4,'Lisa Wilson','lisawilson@example.com');
INSERT INTO "Photographers" VALUES(5,'David Lee','davidlee@example.com');
CREATE TABLE Photographers_Styles (
    photographerId INTEGER REFERENCES Photographers(id),
    styleId INTEGER REFERENCES Styles(id));
INSERT INTO "Photographers_Styles" VALUES(1,1);
INSERT INTO "Photographers_Styles" VALUES(2,1);
INSERT INTO "Photographers_Styles" VALUES(3,2);
INSERT INTO "Photographers_Styles" VALUES(4,2);
INSERT INTO "Photographers_Styles" VALUES(5,3);
INSERT INTO "Photographers_Styles" VALUES(3,1);
INSERT INTO "Photographers_Styles" VALUES(4,1);
INSERT INTO "Photographers_Styles" VALUES(5,1);
CREATE TABLE Styles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    styleName VARCHAR(255));
INSERT INTO "Styles" VALUES(1,'Portrait');
INSERT INTO "Styles" VALUES(2,'Landscape');
INSERT INTO "Styles" VALUES(3,'Wedding');
INSERT INTO "Styles" VALUES(4,'Fashion');
INSERT INTO "Styles" VALUES(5,'Commercial');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Photographers',5);
INSERT INTO "sqlite_sequence" VALUES('Styles',5);
INSERT INTO "sqlite_sequence" VALUES('JobPostings',5);
COMMIT;
