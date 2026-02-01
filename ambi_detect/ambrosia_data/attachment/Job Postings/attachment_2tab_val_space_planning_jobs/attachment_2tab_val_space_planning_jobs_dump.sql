BEGIN TRANSACTION;
CREATE TABLE Employees(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Role TEXT,
    ContactInfo TEXT
);
INSERT INTO "Employees" VALUES(1,'Alice Brown','Interior Designer','alice@example.com');
INSERT INTO "Employees" VALUES(2,'Bob Green','Landscape Architect','bob@example.com');
CREATE TABLE Interior_Designers(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    Email TEXT,
    PhoneNumber TEXT,
    Style VARCHAR(25) 
);
INSERT INTO "Interior_Designers" VALUES(1,'John Doe','123 Main St','john@example.com','555-1234','Minimalism');
INSERT INTO "Interior_Designers" VALUES(2,'Jane Smith','456 Elm St','jane@example.com','555-5678','Modern');
CREATE TABLE Landscape_Architects(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    Email TEXT,
    PhoneNumber TEXT,
    Style VARCHAR(25) 
);
INSERT INTO "Landscape_Architects" VALUES(1,'Michael Johnson','987 Oak St','michael@example.com','555-9012','Minimalism');
INSERT INTO "Landscape_Architects" VALUES(2,'Emily Davis','654 Pine St','emily@example.com','555-1357','Contemporary');
CREATE TABLE Materials(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    ProjectID INTEGER REFERENCES Projects(ID),
    MaterialName TEXT,
    Quantity INTEGER,
    UnitPrice REAL
);
INSERT INTO "Materials" VALUES(1,1,'Wood Flooring',100,5.0);
INSERT INTO "Materials" VALUES(2,2,'Stone Tiles',50,7.0);
CREATE TABLE Projects(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    DesignerID INTEGER REFERENCES Interior_Designers(ID),
    ArchitectID INTEGER REFERENCES Landscape_Architects(ID),
    Title TEXT,
    Location TEXT,
    StartDate DATE,
    EndDate DATE,
    Budget REAL,
    Status TEXT 
);
INSERT INTO "Projects" VALUES(1,1,NULL,'Living Room Redesign','Los Angeles','2022-01-01','2022-03-31',10000.0,'Completed');
INSERT INTO "Projects" VALUES(2,NULL,2,'Garden Pathway','San Francisco','2022-02-01','2022-04-30',7500.0,'In Progress');
CREATE TABLE Staffing(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    ProjectID INTEGER REFERENCES Projects(ID),
    EmployeeID INTEGER REFERENCES Employees(ID),
    HoursWorked INT,
    PayRate DECIMAL(10,2)
);
INSERT INTO "Staffing" VALUES(1,1,1,40,25);
INSERT INTO "Staffing" VALUES(2,2,2,50,27);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Interior_Designers',2);
INSERT INTO "sqlite_sequence" VALUES('Landscape_Architects',2);
INSERT INTO "sqlite_sequence" VALUES('Projects',2);
INSERT INTO "sqlite_sequence" VALUES('Materials',2);
INSERT INTO "sqlite_sequence" VALUES('Employees',2);
INSERT INTO "sqlite_sequence" VALUES('Staffing',2);
COMMIT;
