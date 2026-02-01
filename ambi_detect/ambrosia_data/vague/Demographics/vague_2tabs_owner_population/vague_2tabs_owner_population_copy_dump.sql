BEGIN TRANSACTION;
CREATE TABLE Address(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Street TEXT,
    City TEXT,
    PersonId INTEGER,
    FOREIGN KEY(PersonId) REFERENCES Person(ID)
);
INSERT INTO "Address" VALUES(1,'123 Main St','New York',1);
INSERT INTO "Address" VALUES(2,'456 Elm St','Los Angeles',2);
INSERT INTO "Address" VALUES(3,'789 Oak St','Chicago',3);
INSERT INTO "Address" VALUES(4,'321 Pine St','Dallas',4);
INSERT INTO "Address" VALUES(5,'654 Maple Ave','Miami',5);
CREATE TABLE EmploymentStatus(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Status TEXT,
    PersonId INTEGER,
    FOREIGN KEY(PersonId) REFERENCES Person(ID)
);
INSERT INTO "EmploymentStatus" VALUES(1,'Unemployed',1);
INSERT INTO "EmploymentStatus" VALUES(2,'Self-Employed',2);
INSERT INTO "EmploymentStatus" VALUES(3,'Full Time',3);
INSERT INTO "EmploymentStatus" VALUES(4,'Part Time',4);
INSERT INTO "EmploymentStatus" VALUES(5,'Retired',5);
CREATE TABLE "Homeownership"(
    Year INTEGER,
    NumberOfHouseholds INT ,
    Percentage FLOAT ,
    FOREIGN KEY(Year) REFERENCES NumbersOfOwners(Year),
    FOREIGN KEY(Percentage) REFERENCES "Owner_Population"(Percentage)
);
INSERT INTO "Homeownership" VALUES(2021,1500000,55.0);
INSERT INTO "Homeownership" VALUES(2022,1550000,56.0);
INSERT INTO "Homeownership" VALUES(2023,1600000,57.0);
INSERT INTO "Homeownership" VALUES(2024,1650000,58.0);
INSERT INTO "Homeownership" VALUES(2025,1700000,59.0);
CREATE TABLE IncomeLevel(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Level TEXT,
    PersonId INTEGER,
    FOREIGN KEY(PersonId) REFERENCES Person(ID)
);
INSERT INTO "IncomeLevel" VALUES(1,'Low',1);
INSERT INTO "IncomeLevel" VALUES(2,'Medium',2);
INSERT INTO "IncomeLevel" VALUES(3,'High',3);
INSERT INTO "IncomeLevel" VALUES(4,'Very High',4);
INSERT INTO "IncomeLevel" VALUES(5,'Ultra High',5);
CREATE TABLE "Owner_Population"(
    Year INTEGER,
    Percentage FLOAT ,
    TotalPopulation INT ,
    PRIMARY KEY(Year)
);
INSERT INTO "Owner_Population" VALUES(2021,55.0,1000000);
INSERT INTO "Owner_Population" VALUES(2022,56.0,1000000);
INSERT INTO "Owner_Population" VALUES(2023,57.0,1000000);
INSERT INTO "Owner_Population" VALUES(2024,58.0,1000000);
INSERT INTO "Owner_Population" VALUES(2025,59.0,1000000);
CREATE TABLE Person(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INT 
);
INSERT INTO "Person" VALUES(1,'John Doe',30);
INSERT INTO "Person" VALUES(2,'Jane Smith',28);
INSERT INTO "Person" VALUES(3,'Michael Johnson',35);
INSERT INTO "Person" VALUES(4,'Emily Davis',26);
INSERT INTO "Person" VALUES(5,'David Brown',40);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Person',5);
INSERT INTO "sqlite_sequence" VALUES('Address',5);
INSERT INTO "sqlite_sequence" VALUES('EmploymentStatus',5);
INSERT INTO "sqlite_sequence" VALUES('IncomeLevel',5);
COMMIT;
