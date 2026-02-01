BEGIN TRANSACTION;
CREATE TABLE Aircrafts (
    id INTEGER PRIMARY KEY,
    Model TEXT,
    Manufacturer TEXT,
    Year_Manufactured INT
);
INSERT INTO "Aircrafts" VALUES(1,'Boeing 737','Boeing',1967);
INSERT INTO "Aircrafts" VALUES(2,'Airbus A320','Airbus',1988);
CREATE TABLE Airlines (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Headquarters TEXT,
    Fleet_Size INT
);
INSERT INTO "Airlines" VALUES(1,'American Airlines','Fort Worth, TX',953);
INSERT INTO "Airlines" VALUES(2,'Delta Airlines','Atlanta, GA',755);
CREATE TABLE Aviation_Academies (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Years_Operating INT
);
INSERT INTO "Aviation_Academies" VALUES(1,'Aviation Academy','New York',25);
INSERT INTO "Aviation_Academies" VALUES(2,'Global Aero Institute','Los Angeles',10);
CREATE TABLE Flight_Attendants (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Experience_Years REAL,
    Training_Center TEXT,
    FOREIGN KEY(Training_Center) REFERENCES Aviation_Academies(Name)
);
INSERT INTO "Flight_Attendants" VALUES(1,'John Doe',30,8.5,'Aviation Academy');
INSERT INTO "Flight_Attendants" VALUES(2,'Jane Smith',27,6.5,'Global Aero Institute');
CREATE TABLE Flight_Routes (
    id INTEGER PRIMARY KEY,
    Route_Number TEXT,
    Origin_Airport TEXT,
    Destination_Airport TEXT,
    Distance REAL
);
INSERT INTO "Flight_Routes" VALUES(1,'AA123','DFW','LAX',2000.0);
INSERT INTO "Flight_Routes" VALUES(2,'DL456','ATL','MIA',1500.0);
CREATE TABLE Pilots (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Licenses TEXT[],
    Training_Center TEXT,
    FOREIGN KEY(Training_Center) REFERENCES Aviation_Academies(Name)
);
INSERT INTO "Pilots" VALUES(1,'Michael Johnson',32,'{ATPL, CPL}','Aviation Academy');
INSERT INTO "Pilots" VALUES(2,'Emily Brown',31,'{ATPL, CPL}','World Flight School');
COMMIT;
