BEGIN TRANSACTION;
CREATE TABLE Cultural_Centers(
    centerID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Cultural_Centers" VALUES(1,'Art Center','New York');
INSERT INTO "Cultural_Centers" VALUES(2,'Theater House','Los Angeles');
INSERT INTO "Cultural_Centers" VALUES(3,'Music Hall','Chicago');
INSERT INTO "Cultural_Centers" VALUES(4,'Dance Studio','Miami');
INSERT INTO "Cultural_Centers" VALUES(5,'Concert Venue','Houston');
CREATE TABLE Cultural_Centers_Performances(
    culturalCenterID INTEGER,
    performanceID INTEGER,
    FOREIGN KEY(culturalCenterID) REFERENCES Cultural_Centers(centerID),
    FOREIGN KEY(performanceID) REFERENCES Performances(performanceID));
INSERT INTO "Cultural_Centers_Performances" VALUES(1,1);
INSERT INTO "Cultural_Centers_Performances" VALUES(1,2);
INSERT INTO "Cultural_Centers_Performances" VALUES(2,3);
INSERT INTO "Cultural_Centers_Performances" VALUES(3,4);
INSERT INTO "Cultural_Centers_Performances" VALUES(4,5);
INSERT INTO "Cultural_Centers_Performances" VALUES(2,1);
INSERT INTO "Cultural_Centers_Performances" VALUES(3,1);
INSERT INTO "Cultural_Centers_Performances" VALUES(4,1);
INSERT INTO "Cultural_Centers_Performances" VALUES(5,1);
CREATE TABLE Employees(
    employeeID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT);
INSERT INTO "Employees" VALUES(1,'John','Smith');
INSERT INTO "Employees" VALUES(2,'Jane','Doe');
INSERT INTO "Employees" VALUES(3,'Robert','Brown');
INSERT INTO "Employees" VALUES(4,'Alice','Green');
INSERT INTO "Employees" VALUES(5,'Tom','White');
CREATE TABLE Managers(
    managerID INTEGER PRIMARY KEY AUTOINCREMENT,
    employeeID INTEGER,
    FOREIGN KEY(employeeID) REFERENCES Employees(employeeID));
INSERT INTO "Managers" VALUES(1,1);
INSERT INTO "Managers" VALUES(2,2);
INSERT INTO "Managers" VALUES(3,3);
INSERT INTO "Managers" VALUES(4,4);
INSERT INTO "Managers" VALUES(5,5);
CREATE TABLE Performances(
    performanceID INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT);
INSERT INTO "Performances" VALUES(1,'Dance_Performances');
INSERT INTO "Performances" VALUES(2,'Musical_Performances');
INSERT INTO "Performances" VALUES(3,'Drama_Performances');
INSERT INTO "Performances" VALUES(4,'Comedy_Shows');
INSERT INTO "Performances" VALUES(5,'Opera');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Cultural_Centers',5);
INSERT INTO "sqlite_sequence" VALUES('Performances',5);
INSERT INTO "sqlite_sequence" VALUES('Employees',5);
INSERT INTO "sqlite_sequence" VALUES('Managers',5);
COMMIT;
