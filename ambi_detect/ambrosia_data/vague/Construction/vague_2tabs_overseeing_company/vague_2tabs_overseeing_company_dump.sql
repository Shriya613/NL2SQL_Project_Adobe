BEGIN TRANSACTION;
CREATE TABLE Contractor (
    id INTEGER PRIMARY KEY,
    name TEXT,
    specialty TEXT,
    license_number TEXT
);
INSERT INTO "Contractor" VALUES(1,'Contractor X','Concrete Work','LX001');
INSERT INTO "Contractor" VALUES(2,'Contractor Y','Plumbing','LY002');
CREATE TABLE Development_Project (
    id INTEGER PRIMARY KEY,
    project_name TEXT,
    developer_id INTEGER,
    contractor_id INTEGER,
    FOREIGN KEY(developer_id) REFERENCES Real_Estate_Developer(id),
    FOREIGN KEY(contractor_id) REFERENCES Contractor(id)
);
INSERT INTO "Development_Project" VALUES(1,'High Rise Building',1,1);
INSERT INTO "Development_Project" VALUES(2,'Residential Complex',2,2);
CREATE TABLE Employee (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    position TEXT,
    contractor_id INTEGER,
    FOREIGN KEY(contractor_id) REFERENCES Contractor(id)
);
INSERT INTO "Employee" VALUES(1,'John','Smith','Site Supervisor',1);
INSERT INTO "Employee" VALUES(2,'Jane','Doe','Engineer',2);
CREATE TABLE Materials (
    id INTEGER PRIMARY KEY,
    material_type TEXT,
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    contractor_id INTEGER,
    FOREIGN KEY(contractor_id) REFERENCES Contractor(id)
);
INSERT INTO "Materials" VALUES(1,'Reinforcement Bar',500,10,1);
INSERT INTO "Materials" VALUES(2,'Pipes',300,8,2);
CREATE TABLE Project_Milestone (
    id INTEGER PRIMARY KEY,
    milestone_description TEXT,
    start_date DATE,
    end_date DATE,
    development_project_id INTEGER,
    FOREIGN KEY(development_project_id) REFERENCES Development_Project(id)
);
INSERT INTO "Project_Milestone" VALUES(1,'Foundation Completed','2023-01-01','2023-01-10',1);
INSERT INTO "Project_Milestone" VALUES(2,'Framing Started','2023-01-11','2023-01-20',1);
CREATE TABLE Real_Estate_Developer (
    id INTEGER PRIMARY KEY,
    name TEXT,
    contact_info TEXT
);
INSERT INTO "Real_Estate_Developer" VALUES(1,'Company A','+123456789');
INSERT INTO "Real_Estate_Developer" VALUES(2,'Company B','+987654321');
COMMIT;
