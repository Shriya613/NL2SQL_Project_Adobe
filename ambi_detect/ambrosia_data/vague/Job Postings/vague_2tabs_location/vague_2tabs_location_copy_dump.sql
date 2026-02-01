BEGIN TRANSACTION;
CREATE TABLE Application (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_first_name TEXT,
    applicant_last_name TEXT,
    application_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    internship_id INTEGER,
    employee_id INTEGER,
    FOREIGN KEY(internship_id) REFERENCES Internship(id),
    FOREIGN KEY(employee_id) REFERENCES Employee(id)
);
INSERT INTO "Application" VALUES(1,'Alice','Brown','2024-02-19 12:32:36',1,1);
INSERT INTO "Application" VALUES(2,'Charlie','Green','2024-02-19 12:32:36',2,2);
CREATE TABLE Company (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT,
    industry TEXT
);
INSERT INTO "Company" VALUES(1,'TechCorp','Technology');
INSERT INTO "Company" VALUES(2,'HealthCo','Healthcare');
CREATE TABLE Department (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    department_name TEXT,
    company_id INTEGER,
    FOREIGN KEY(company_id) REFERENCES Company(id)
);
INSERT INTO "Department" VALUES(1,'Engineering',1);
INSERT INTO "Department" VALUES(2,'Marketing',1);
CREATE TABLE Employee (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number TEXT,
    hire_date DATE,
    department_id INTEGER,
    FOREIGN KEY(department_id) REFERENCES Department(id)
);
INSERT INTO "Employee" VALUES(1,'John','Doe','john.doe@example.com','555-1234','2022-01-01',1);
INSERT INTO "Employee" VALUES(2,'Jane','Smith','jane.smith@example.com','555-5678','2022-02-01',2);
CREATE TABLE "Internship" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    start_date DATE,
    end_date DATE,
    office_address_id INTEGER,
    FOREIGN KEY(office_address_id) REFERENCES "Location"(id));
INSERT INTO "Internship" VALUES(1,'Software Engineer','Develop and maintain software applications.','2022-07-01','2023-08-31',1);
INSERT INTO "Internship" VALUES(2,'Data Analyst','Analyze data to support business decisions.','2022-09-01','2023-12-31',2);
CREATE TABLE "Location" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT,
    postal_code TEXT
);
INSERT INTO "Location" VALUES(1,'123 Main St','60601');
INSERT INTO "Location" VALUES(2,'456 Elm St','60602');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Location',2);
INSERT INTO "sqlite_sequence" VALUES('Company',2);
INSERT INTO "sqlite_sequence" VALUES('Department',2);
INSERT INTO "sqlite_sequence" VALUES('Employee',2);
INSERT INTO "sqlite_sequence" VALUES('Application',2);
INSERT INTO "sqlite_sequence" VALUES('Internship',2);
COMMIT;
