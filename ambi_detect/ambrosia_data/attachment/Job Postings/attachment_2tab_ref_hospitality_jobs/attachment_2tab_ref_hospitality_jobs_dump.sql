BEGIN TRANSACTION;
CREATE TABLE Agencies (
    AgencyID INTEGER PRIMARY KEY,
    Name TEXT
);
INSERT INTO "Agencies" VALUES(1,'Expedia');
INSERT INTO "Agencies" VALUES(2,'ABC Travel');
CREATE TABLE BookedTrips (
    TripID INTEGER PRIMARY KEY,
    StartDate DATE,
    EndDate DATE,
    NumOfPeople INT,
    TotalCost REAL,
    AgentID INTEGER,
    GuideID INTEGER,
    FOREIGN KEY(AgentID) REFERENCES TravelAgents(AgentID),
    FOREIGN KEY(GuideID) REFERENCES TourGuides(GuideID)
);
INSERT INTO "BookedTrips" VALUES(1,'2021-07-01','2021-07-10',4,4000.0,1,1);
INSERT INTO "BookedTrips" VALUES(2,'2021-08-01','2021-08-10',6,6000.0,2,2);
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT,
    PhoneNumber CHAR(10),
    TourGUIDEID INTEGER,
    FOREIGN KEY(TourGUIDEID) REFERENCES TourGuides(GuideID)
);
INSERT INTO "Customers" VALUES(1,'Alice','Smith','alice@example.com','1234567890',1);
INSERT INTO "Customers" VALUES(2,'Bob','Johnson','bob@example.com','0987654321',2);
CREATE TABLE TourGuides (
    GuideID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Experience INT,
    AgencyID INTEGER,
    FOREIGN KEY(AgencyID) REFERENCES Agencies(AgencyID)
);
INSERT INTO "TourGuides" VALUES(1,'John Doe',30,8,1);
INSERT INTO "TourGuides" VALUES(2,'Jane Smith',29,7,2);
CREATE TABLE TravelAgents (
    AgentID INTEGER PRIMARY KEY,
    Name TEXT,
    Salary REAL,
    CommissionRate DECIMAL(5,2),
    AgencyID INTEGER,
    FOREIGN KEY(AgencyID) REFERENCES Agencies(AgencyID)
);
INSERT INTO "TravelAgents" VALUES(1,'Mike Johnson',60000.0,0.1,1);
INSERT INTO "TravelAgents" VALUES(2,'Emily Brown',55000.0,0.12,2);
COMMIT;
