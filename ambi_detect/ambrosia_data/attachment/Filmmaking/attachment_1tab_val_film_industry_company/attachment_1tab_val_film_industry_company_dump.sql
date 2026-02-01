BEGIN TRANSACTION;
CREATE TABLE Actors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    First_Name TEXT,
    Last_Name TEXT,
    Date_of_Birth DATE,
    Nationality TEXT);
INSERT INTO "Actors" VALUES(1,'Tom','Hanks','1956-07-09','USA');
INSERT INTO "Actors" VALUES(2,'Meryl','Streep','1949-06-22','USA');
CREATE TABLE Companies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Film_Industry_Type TEXT ,
    Number_Of_Employees INT,
    Annual_Revenue REAL,
    Founded DATE);
INSERT INTO "Companies" VALUES(1,'Big Studios','Production Company',500,123456789.0,'2000-01-01');
INSERT INTO "Companies" VALUES(2,'Little Movies','Production Company',300,543212345.0,'2010-01-01');
INSERT INTO "Companies" VALUES(3,'Global Distribs','Distribution Company',500,654321234.0,'2005-01-01');
INSERT INTO "Companies" VALUES(4,'Small Screen','Distribution Company',400,765432123.0,'2015-01-01');
CREATE TABLE Directors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Birth_Place TEXT,
    Years_Experience INT);
INSERT INTO "Directors" VALUES(1,'John Doe','New York, USA',20);
INSERT INTO "Directors" VALUES(2,'Jane Smith','London, UK',15);
CREATE TABLE Films(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Director_Name TEXT,
    Production_Companies TEXT,
    Distribution_Companies TEXT,
    Release_Date DATE,
    Running_Time INT,
    Genre TEXT,
    Box_Office_Earnings REAL);
INSERT INTO "Films" VALUES(1,'Blockbuster Movie','John Doe','Big Studios','Global Distribs','2021-01-01',120,'Action',100000000.0);
INSERT INTO "Films" VALUES(2,'Art House Film','Jane Smith','Little Movies','Small Screen','2021-06-01',90,'Drama',5000000.0);
CREATE TABLE Producers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Birth_Place TEXT,
    Total_Movies_Produced INT);
INSERT INTO "Producers" VALUES(1,'Alice Johnson','Los Angeles, USA',10);
INSERT INTO "Producers" VALUES(2,'Bob Brown','Toronto, Canada',15);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Companies',4);
INSERT INTO "sqlite_sequence" VALUES('Films',2);
INSERT INTO "sqlite_sequence" VALUES('Directors',2);
INSERT INTO "sqlite_sequence" VALUES('Producers',2);
INSERT INTO "sqlite_sequence" VALUES('Actors',2);
COMMIT;
