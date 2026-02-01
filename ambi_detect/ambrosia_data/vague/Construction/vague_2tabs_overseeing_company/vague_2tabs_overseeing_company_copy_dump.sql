BEGIN TRANSACTION;
CREATE TABLE "Development_Project" (
    id INTEGER PRIMARY KEY,
    project_name TEXT,
    developer_id INTEGER,
    FOREIGN KEY(developer_id) REFERENCES "Overseeing_Company"(id));
INSERT INTO "Development_Project" VALUES(1,'High Rise Building',1);
INSERT INTO "Development_Project" VALUES(2,'Residential Complex',2);
CREATE TABLE "Employee" (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    position TEXT);
INSERT INTO "Employee" VALUES(1,'John','Smith','Site Supervisor');
INSERT INTO "Employee" VALUES(2,'Jane','Doe','Engineer');
CREATE TABLE "Materials" (
    id INTEGER PRIMARY KEY,
    material_type TEXT,
    quantity INTEGER,
    unit_price DECIMAL(10,2));
INSERT INTO "Materials" VALUES(1,'Reinforcement Bar',500,10);
INSERT INTO "Materials" VALUES(2,'Pipes',300,8);
CREATE TABLE "Overseeing_Company" (
    id INTEGER PRIMARY KEY,
    name TEXT,
    contact_info TEXT
);
INSERT INTO "Overseeing_Company" VALUES(1,'Company A','+123456789');
INSERT INTO "Overseeing_Company" VALUES(2,'Company B','+987654321');
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
COMMIT;
