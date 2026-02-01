BEGIN TRANSACTION;
CREATE TABLE Employees (
    employeeID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    jobTitle TEXT,
    salary REAL);
INSERT INTO "Employees" VALUES(1,'Alan','Smith','Carpenter',60000.0);
INSERT INTO "Employees" VALUES(2,'Beth','Davis','Electrician',65000.0);
INSERT INTO "Employees" VALUES(3,'Charlie','Martin','Plumber',62000.0);
INSERT INTO "Employees" VALUES(4,'David','Parker','Site Supervisor',70000.0);
INSERT INTO "Employees" VALUES(5,'Eva','Collins','Architect',90000.0);
CREATE TABLE Equipment (
    equipmentID INTEGER PRIMARY KEY,
    type TEXT,
    make TEXT,
    model TEXT,
    purchasePrice REAL);
INSERT INTO "Equipment" VALUES(1,'Excavator','Caterpillar','320D',75000.0);
INSERT INTO "Equipment" VALUES(2,'Crane','Manitowoc','Model 222',120000.0);
INSERT INTO "Equipment" VALUES(3,'Forklift','Hyster','FT30-40',30000.0);
INSERT INTO "Equipment" VALUES(4,'Scaffolding','Rosendahl','RSC 500',10000.0);
INSERT INTO "Equipment" VALUES(5,'Truck','Ford','F-350',35000.0);
CREATE TABLE Materials (
    materialID INTEGER PRIMARY KEY,
    name TEXT,
    quantity REAL,
    unitCost REAL);
INSERT INTO "Materials" VALUES(1,'Concrete',100.0,50.0);
INSERT INTO "Materials" VALUES(2,'Steel Rebars',50.0,60.0);
INSERT INTO "Materials" VALUES(3,'Windows',150.0,150.0);
INSERT INTO "Materials" VALUES(4,'Doors',75.0,200.0);
INSERT INTO "Materials" VALUES(5,'Roofing Materials',100.0,80.0);
CREATE TABLE Projects (
    projectID INTEGER PRIMARY KEY,
    projectName TEXT,
    startDate DATE,
    endDate DATE,
    budget REAL,
    Project_Manager TEXT,
    Construction_Manager TEXT,
    Edifice TEXT);
INSERT INTO "Projects" VALUES(1,'New Office Building','2023-01-01','2023-12-31',1000000.0,'John Doe','Jane Smith','High Rise');
INSERT INTO "Projects" VALUES(2,'School Renovation','2023-02-01','2023-11-30',800000.0,'Mike Johnson','Lisa Brown','Elementary School');
INSERT INTO "Projects" VALUES(3,'Warehouse Expansion','2023-03-01','2023-12-15',750000.0,'Mark Williams','Karen Lee','Industrial Park');
CREATE TABLE Suppliers (
    supplierID INTEGER PRIMARY KEY,
    companyName TEXT,
    contactName TEXT,
    phoneNumber TEXT,
    address TEXT);
INSERT INTO "Suppliers" VALUES(1,'ABC Concrete','Tom Anderson','555-123-4567','123 Main St, Concrete City');
INSERT INTO "Suppliers" VALUES(2,'XYZ Steel','Jessica Thomas','555-789-0123','456 Second Ave, Steelville');
INSERT INTO "Suppliers" VALUES(3,'Window Manufacturers Inc.','George Wilson','555-321-6540','789 Third St, Windowtown');
INSERT INTO "Suppliers" VALUES(4,'Door Makers Ltd.','Betty Clark','555-987-6544','135 Fourth St, Doorway');
INSERT INTO "Suppliers" VALUES(5,'Roofing Solutions','Samuel Harris','555-555-5555','987 Fifth St, Roofsville');
CREATE TABLE Tasks (
    taskID INTEGER PRIMARY KEY,
    description TEXT,
    status TEXT ,
    startTime TIMESTAMP,
    completionTime TIMESTAMP,
    assignedEmployeeID INTEGER,
    FOREIGN KEY(assignedEmployeeID) REFERENCES Employees(employeeID));
INSERT INTO "Tasks" VALUES(1,'Prepare site for construction','In Progress','2023-01-02 08:00:00','2023-01-02 17:00:00',1);
INSERT INTO "Tasks" VALUES(2,'Set up temporary office','Completed','2023-01-03 09:00:00','2023-01-03 16:00:00',2);
INSERT INTO "Tasks" VALUES(3,'Install electrical system','Not Started',NULL,NULL,3);
INSERT INTO "Tasks" VALUES(4,'Install plumbing','In Progress','2023-01-04 08:00:00','2023-01-04 17:00:00',4);
INSERT INTO "Tasks" VALUES(5,'Begin framing of building','In Progress','2023-01-05 08:00:00','2023-01-05 17:00:00',5);
COMMIT;
