BEGIN TRANSACTION;
CREATE TABLE Company(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100),
    contact_number TEXT,
    address TEXT,
    FOREIGN KEY(id) REFERENCES Trade_Jobs(company_owner)
);
INSERT INTO "Company" VALUES(1,'ABC Construction','+123456789','123 Main St');
INSERT INTO "Company" VALUES(2,'XYZ Electricians','+987654321','456 Elm St');
CREATE TABLE Education(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    degree VARCHAR(100),
    major VARCHAR(100),
    institution TEXT,
    completion_year YEAR,
    FOREIGN KEY(degree) REFERENCES Employee(position)
);
INSERT INTO "Education" VALUES(1,'Project Manager','Business Administration','State University',2017);
INSERT INTO "Education" VALUES(2,'Marketing Specialist','Marketing','City College',2018);
CREATE TABLE Employee(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_name VARCHAR(100),
    position VARCHAR(50),
    hire_date DATE,
    FOREIGN KEY(employee_name) REFERENCES Trade_Jobs(trade_job)
);
INSERT INTO "Employee" VALUES(1,'Alice Jones','Manager','2018-01-01');
INSERT INTO "Employee" VALUES(2,'Bob Smith','Supervisor','2019-02-01');
CREATE TABLE Trade_Jobs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_job VARCHAR(100),
    company_owner VARCHAR(100),
    posted_date DATE,
    salary REAL
);
INSERT INTO "Trade_Jobs" VALUES(1,'Construction Worker','Derek Brown','2023-01-01',40000.0);
INSERT INTO "Trade_Jobs" VALUES(2,'Construction Worker','John Smith','2023-01-02',45000.0);
INSERT INTO "Trade_Jobs" VALUES(3,'Electrician','Derek Brown','2023-02-01',60000.0);
INSERT INTO "Trade_Jobs" VALUES(4,'Electrician','Mary Johnson','2023-02-02',65000.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Trade_Jobs',4);
INSERT INTO "sqlite_sequence" VALUES('Company',2);
INSERT INTO "sqlite_sequence" VALUES('Employee',2);
INSERT INTO "sqlite_sequence" VALUES('Education',2);
COMMIT;
