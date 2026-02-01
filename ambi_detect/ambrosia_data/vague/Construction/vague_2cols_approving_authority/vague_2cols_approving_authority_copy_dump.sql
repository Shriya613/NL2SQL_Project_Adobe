BEGIN TRANSACTION;
CREATE TABLE "Boards"(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Approving_Authority TEXT,
    EngineeringProject TEXT);
INSERT INTO "Boards" VALUES(1,'City Planning Department','Highway Expansion Project');
INSERT INTO "Boards" VALUES(2,'Department of Transportation','Railway Modernization Project');
INSERT INTO "Boards" VALUES(3,'Environment Protection Agency','Industrial Waste Treatment Plant');
INSERT INTO "Boards" VALUES(4,'Health & Safety Executive','Hospital Construction Project');
INSERT INTO "Boards" VALUES(5,'Local Council','Historic Building Restoration Project');
CREATE TABLE ConstructionSite(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    SiteName TEXT,
    Location TEXT,
    StartDate DATE,
    EndDate DATE,
    Status TEXT ,
    ForeignKeyId INTEGER);
CREATE TABLE ContractorDetails(
    Id INT PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    PhoneNumber TEXT,
    EmailAddress TEXT,
    ServicesOffered TEXT);
INSERT INTO "ContractorDetails" VALUES(1,'ABC Construction','123 Main St, Cityville','+1-123-456-7890','abc@construction.com','Residential and Commercial Building');
INSERT INTO "ContractorDetails" VALUES(2,'XYZ Infrastructure','456 Elm St, Cityville','+1-123-456-7891','xyz@infrastructure.com','Infrastructure Development');
INSERT INTO "ContractorDetails" VALUES(3,'DEF Electricals','789 Oak St, Cityville','+1-123-456-7892','def@electricals.com','Electrical Works');
INSERT INTO "ContractorDetails" VALUES(4,'GHI Plumbing','135 Maple St, Cityville','+1-123-456-7893','ghi@plumbing.com','Plumbing Services');
INSERT INTO "ContractorDetails" VALUES(5,'JKL Landscaping','456 Pine St, Cityville','+1-123-456-7894','jkl@landscaping.com','Landscape Design and Installation');
CREATE TABLE Employees(
    Id INT PRIMARY KEY,
    Name TEXT,
    JobTitle TEXT,
    ContactInfo TEXT,
    HireDate DATE,
    Salary REAL);
INSERT INTO "Employees" VALUES(1,'John Doe','Carpenter','+1-987-654-3210','2021-01-15',50000.0);
INSERT INTO "Employees" VALUES(2,'Jane Smith','Electrician','+1-987-654-3211','2020-06-01',55000.0);
INSERT INTO "Employees" VALUES(3,'Alice Johnson','Plumber','+1-987-654-3212','2019-08-10',52000.0);
INSERT INTO "Employees" VALUES(4,'Bob Brown','Foreman','+1-987-654-3213','2018-11-15',60000.0);
INSERT INTO "Employees" VALUES(5,'Charlie Davis','Supervisor','+1-987-654-3214','2017-05-01',65000.0);
CREATE TABLE Equipment(
    Id INT PRIMARY KEY,
    Type TEXT,
    ModelNumber TEXT,
    PurchaseDate DATE,
    CostPrice REAL,
    MarketValue REAL);
INSERT INTO "Equipment" VALUES(1,'Excavator','EX1000','2021-02-10',100000.0,120000.0);
INSERT INTO "Equipment" VALUES(2,'Bulldozer','BD2000','2020-09-01',150000.0,180000.0);
INSERT INTO "Equipment" VALUES(3,'Crane','CR3000','2019-05-05',200000.0,250000.0);
INSERT INTO "Equipment" VALUES(4,'Dumper Truck','DT4000','2018-07-10',50000.0,65000.0);
INSERT INTO "Equipment" VALUES(5,'Concrete Mixer','CM5000','2017-10-01',30000.0,40000.0);
CREATE TABLE ProjectStatus(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    ProjectId INTEGER,
    Date DATE,
    TaskCompleted TEXT,
    BudgetSpent REAL,
    ExpectedCompletionDate DATE,
    FOREIGN KEY(ProjectId) REFERENCES ConstuctionSite(Id));
INSERT INTO "ProjectStatus" VALUES(1,1,'2021-06-15','Foundation Laid',20000.0,'2021-07-15');
INSERT INTO "ProjectStatus" VALUES(2,1,'2021-07-15','Walls Erected',35000.0,'2021-08-15');
INSERT INTO "ProjectStatus" VALUES(3,1,'2021-08-15','Roof Installed',50000.0,'2021-09-15');
INSERT INTO "ProjectStatus" VALUES(4,1,'2021-09-15','Interior Finished',65000.0,'2021-10-15');
INSERT INTO "ProjectStatus" VALUES(5,1,'2021-10-15','Final Inspection',75000.0,'2021-11-15');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('ProjectStatus',5);
INSERT INTO "sqlite_sequence" VALUES('Boards',5);
COMMIT;
