BEGIN TRANSACTION;
CREATE TABLE Crude_Death_Rate (
    year INTEGER,
    rate REAL,
    FOREIGN KEY(year) REFERENCES Province(year),
    PRIMARY KEY(year)
);
INSERT INTO "Crude_Death_Rate" VALUES(2000,9.2);
INSERT INTO "Crude_Death_Rate" VALUES(2001,9.3);
INSERT INTO "Crude_Death_Rate" VALUES(2002,9.4);
INSERT INTO "Crude_Death_Rate" VALUES(2003,9.5);
INSERT INTO "Crude_Death_Rate" VALUES(2004,9.6);
CREATE TABLE Disease (
    diseaseID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    deathrate REAL 
);
INSERT INTO "Disease" VALUES(1,'Heart Disease',0.32);
INSERT INTO "Disease" VALUES(2,'Cancer',0.25);
INSERT INTO "Disease" VALUES(3,'Respiratory diseases',0.07);
INSERT INTO "Disease" VALUES(4,'Stroke',0.1);
INSERT INTO "Disease" VALUES(5,'Diabetes',0.05);
CREATE TABLE Number_Of_Deaths (
    year INTEGER,
    total_deaths INTEGER,
    FOREIGN KEY(year) REFERENCES Province(year),
    PRIMARY KEY(year)
);
INSERT INTO "Number_Of_Deaths" VALUES(2000,50000);
INSERT INTO "Number_Of_Deaths" VALUES(2001,51000);
INSERT INTO "Number_Of_Deaths" VALUES(2002,52000);
INSERT INTO "Number_Of_Deaths" VALUES(2003,53000);
INSERT INTO "Number_Of_Deaths" VALUES(2004,54000);
CREATE TABLE Person (
    personID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    dateofbirth DATE 
);
INSERT INTO "Person" VALUES(1,'John Doe','1980-01-02');
INSERT INTO "Person" VALUES(2,'Jane Smith','1975-03-15');
INSERT INTO "Person" VALUES(3,'Alice Brown','1990-06-27');
INSERT INTO "Person" VALUES(4,'Bob Johnson','1985-08-10');
INSERT INTO "Person" VALUES(5,'Charlie White','1978-12-12');
CREATE TABLE Province (
    provinceID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    year INTEGER,
    population INTEGER ,
    FOREIGN KEY(year) REFERENCES Number_Of_Deaths(year)
);
INSERT INTO "Province" VALUES(1,'New York',2000,10000000);
INSERT INTO "Province" VALUES(2,'California',2001,30000000);
INSERT INTO "Province" VALUES(3,'Texas',2002,20000000);
INSERT INTO "Province" VALUES(4,'Florida',2003,15000000);
INSERT INTO "Province" VALUES(5,'Illinois',2004,10000000);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Person',5);
INSERT INTO "sqlite_sequence" VALUES('Disease',5);
INSERT INTO "sqlite_sequence" VALUES('Province',5);
COMMIT;
